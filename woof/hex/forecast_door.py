# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""The forecast front door: an init and a registered mesh in, history out.

``woof hex forecast`` is the user-reachable command over the same execution
path the measurement harness drives.  It does not reimplement the model: it
binds a REGISTERED mesh through ``src/hexcore/drivers/mpas_mesh_binding.py``, hands the
bound shape to ``src/hexcore/drivers/run_cuda_v841_forecast.py`` -- whose ``execute_forecast``
is the one integration loop this project has -- and owns everything a
measurement harness deliberately does not:

* fail-closed argument resolution, with every refusal naming the wrong
  result it prevents and the command that fixes it;
* device-memory ADMISSION before CUDA is touched, against memory measured at
  the moment of the decision;
* destination discipline (the run's scratch never lands inside its output);
* one run receipt that states what ran, what it claims, and the exact
  ``woof hex render`` command that consumes the output.

WHY ADMISSION IS THE POINT OF THIS DOOR.  The driver's floor and this door
answer from ONE surface, ``woof.hex.device_admission``: the same measured
affine row, the same headroom, the same :func:`required_free_bytes` sum.
Before that surface existed the driver held a separate linear floor scaled
from an asserted 24 GiB constant, which on the published 40,962-cell mesh
admitted at about 6.0 GiB free while the measured peak was 9,948 MiB -- a
card between those numbers passed the floor, spent minutes loading a mesh
and compiling kernels, and then died inside a CuPy allocation part-way
through a run.  That is the concrete breakage this gate prevents.  When the
door admits on a card's OWN measured row it forwards the same requirement to
the driver (``--required-free-bytes``), so the two gates cannot disagree.

WHY THE MODEL IS NOT A FROZEN BUDGET.  Free device memory is read from the
driver at the moment of the decision, never carried from a previous run or a
previous card, because the memory a desktop session is already holding is
part of the answer.  The fitted row is a measurement with a provenance
string, and a card whose own ledger has been run supplies its own row with
``--device-fixed-mib`` / ``--device-bytes-per-cell``.

This module is stdlib-only at import.  numpy, netCDF4 and cupy are pulled
inside the handlers that need them, so ``woof hex --help`` and every
argument refusal work on a box with no CUDA lane at all.
"""

from __future__ import annotations

from .stochastic import resolve_options as resolve_stochastic_options

import argparse
from dataclasses import dataclass, field
import datetime as _dt
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from .device_admission import (
    DEFAULT_HEADROOM_BYTES,
    FOOTPRINT_MODEL,
    REFERENCE_CARD,
    ROLLBACK_ARMED_BYTES_PER_CELL,
    CardProfile,
    FootprintModel,
    ShapedFootprintModel,
    card_profile_from_attributes,
    model_for_card,
    required_free_bytes,
)
from .physics_backend_admission import (
    DEFAULT_BACKEND,
    registered_backend_names,
    resolve_backend,
)
from . import engine_pin
from . import forecast_preset
from . import mpas_defaults
from .host_identity import host_record
from .errors import ConfigurationRefusal, MpasPortError
from . import time_ledger

__all__ = [
    "AdmissionVerdict",
    "CardProfile",
    "DEFAULT_HEADROOM_BYTES",
    "FOOTPRINT_MODEL",
    "FootprintModel",
    "ShapedFootprintModel",
    "read_card_profile",
    "request_configuration",
    "resolve_admission_model",
    "ForecastDoorRefusal",
    "ForecastRequest",
    "MeshRow",
    "PROJECT_ROOT",
    "RECEIPT_SCHEMA",
    "add_forecast_arguments",
    "admission_verdict",
    "admit_timestep",
    "admit_architecture",
    "admit_device",
    "build_receipt",
    "checkout_root",
    "load_registry",
    "measure_device_memory",
    "probe_numeric_route",
    "read_device_compute",
    "read_nvrtc_targets",
    "render_command",
    "require_admitted",
    "resolve_repo",
    "resolve_request",
    "run_forecast",
    "seam_pin_problem",
    "seam_pin_record",
]

MIB = 1024**2

#: ``DEFAULT_HEADROOM_BYTES``, ``FOOTPRINT_MODEL`` and ``FootprintModel``
#: live in :mod:`woof.hex.device_admission` -- the one admission surface
#: the driver's floor also reads -- and are re-exported here unchanged for
#: every existing caller.

#: The receipt this door writes beside the history files.
RECEIPT_SCHEMA = "gpuwm-hex.forecast-run/v1"

#: The receipt the driver writes into the same directory.  Read back and
#: embedded whole, so the door's receipt never restates a driver number.
DRIVER_RECEIPT_NAME = "cuda-v841-forecast-receipt.json"

class ForecastDoorRefusal(MpasPortError):
    """A named forecast-door refusal: what breaks, then the remedy."""


def _refuse(message: str) -> "ForecastDoorRefusal":
    return ForecastDoorRefusal(message)


def _mib(value: float) -> str:
    return f"{value / MIB:,.1f} MiB"


# ---------------------------------------------------------------------------
# where the drivers live
# ---------------------------------------------------------------------------
def checkout_root(start: Path | None = None) -> Path | None:
    """The source checkout this package was imported from, or ``None``.

    ``parents[2]`` is the project root of a ``src/`` layout checkout; from an
    installed wheel it is whatever directory happens to sit above
    ``site-packages``.  Only the CLI's mesh defaults (``data/meshes``) and
    ``version``'s report use it now: the forecast drivers ship inside the
    package and no door needs a checkout to open.
    """

    root = (start or Path(__file__).resolve().parents[2])
    if (root / "src" / "hexcore").is_dir() and (root / "tools").is_dir():
        return root
    return None


PROJECT_ROOT = checkout_root()

#: The forecast drivers, inside the package that is executing.  They lived
#: in the repository's ``tools/`` through 0.3.1, which a wheel does not
#: carry, so ``woof hex forecast`` refused on every install with "needs the
#: woof hex SOURCE CHECKOUT" -- the forecast was not reachable from
#: ``pip install woof hex``.  Loaded from here, the drivers and the modules
#: they verify by SHA-256 are always one copy.
DRIVERS_DIR = Path(__file__).resolve().parent / "drivers"

#: What the drivers directory must carry.  Named individually so a damaged
#: install says WHICH file is absent.
_REQUIRED_DRIVERS = (
    "run_cuda_v841_full_physics_x4.py",
    "run_cuda_v841_forecast.py",
    "mpas_mesh_binding.py",
)


def resolve_repo(explicit: Path | None) -> Path:
    """The directory the forecast drivers are loaded from: the package's own.

    ``--repo`` named a woof hex checkout's ``tree/`` through 0.3.1, because
    the drivers lived in its ``tools/``.  They ship in the package now, and
    loading a checkout's copy beside an installed package is the hybrid in
    which a gate hashes one copy of a module while another executes, so the
    value is no longer used.  It is still accepted, so existing scripts keep
    working, and the door says it was not used.
    """

    if explicit is not None:
        print(
            f"NOTE --repo {explicit} is not used: the forecast drivers ship "
            f"in the package ({DRIVERS_DIR}) since 0.3.2.",
            file=sys.stderr,
            flush=True,
        )
    missing = [name for name in _REQUIRED_DRIVERS if not (DRIVERS_DIR / name).is_file()]
    if missing:
        raise _refuse(
            f"this woof hex install is missing {', '.join(missing)} under "
            f"{DRIVERS_DIR}; the forecast drivers ship in the package, so the "
            "install is damaged.  Reinstall recast-woof."
        )
    return DRIVERS_DIR


def _load_module(name: str, path: Path):
    """Load one driver by path, the way the registered-mesh runner does."""

    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:  # pragma: no cover
        raise _refuse(f"{path} could not be loaded as a Python module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# the mesh registry, read rather than restated
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshRow:
    """One registered mesh, reduced to what the door decides with."""

    name: str
    cells: int
    dt_seconds: float
    #: The row's declared nominal spacing, in metres.  Carried because the
    #: convection default (off below 3 km) is answerable
    #: from the row alone -- no card, no file -- exactly as the timestep
    #: anchor is, and preflight's answer and the run's must not disagree.
    #: ``MeshBinding`` requires it, so :func:`load_registry` always fills it;
    #: the ``None`` default exists for constructed doubles, and a row that
    #: offers no spacing is recorded as ``source: "unknown-spacing"`` rather
    #: than having one guessed for it.
    nominal_dx_m: float | None = None
    #: The native invariant stream's file name the row declares
    #: (``MeshBinding.invariant_file``), ``None`` for a row whose workflow
    #: configures none.  :func:`resolve_invariant` looks for it beside the
    #: grid and static.
    invariant_file: str | None = None
    #: A generated variable-resolution row's background (coarsest) spacing,
    #: in metres, ``None`` on a uniform row.  Preflight hands it to the
    #: convection decision so a variable mesh is decided per cell from the
    #: row alone, as the bind decides it from the grid.
    coarsest_dx_m: float | None = None


def load_registry(drivers: Path | None = None) -> dict[str, MeshRow]:
    """The registered meshes, read out of ``woof/hex/drivers/mpas_mesh_binding.py``.

    Read, never restated.  A second copy of the cell counts in this file
    would be a second thing to keep true, and the number it would drift on
    is the one every admission decision divides by.  ``drivers`` is the
    directory :func:`resolve_repo` returns; the default is the package's.
    """

    module = _load_module(
        "mpas_mesh_binding", Path(drivers or DRIVERS_DIR) / "mpas_mesh_binding.py"
    )
    return {
        name: MeshRow(
            name=name,
            cells=int(row.n_cells),
            dt_seconds=float(row.dt_seconds),
            nominal_dx_m=float(row.nominal_dx_m),
            invariant_file=getattr(row, "invariant_file", None),
            coarsest_dx_m=getattr(row, "coarsest_dx_m", None),
        )
        for name, row in module.MESH_BINDINGS.items()
    }


# ---------------------------------------------------------------------------
# the fitted footprint, and the decision it feeds
# ---------------------------------------------------------------------------
#: The instrument that produces a row for a DIFFERENT card.  Named in the
#: refusal, because "your card may be smaller" is only useful beside the
#: command that settles it.
LEDGER_PROBE = "tools/device_memory_ledger/hex_ledger_probe.py"


@dataclass(frozen=True)
class AdmissionVerdict:
    """One admission decision, with every number it was made from."""

    admitted: bool
    mesh: str
    cells: int
    predicted_bytes: int
    headroom_bytes: int
    required_bytes: int
    free_bytes: int
    total_bytes: int
    shortfall_bytes: int
    fitted_cells: int
    alternatives: tuple[str, ...]
    model_provenance: str
    model: ShapedFootprintModel | None = None
    stochastic_basis_bytes: int = 0
    stochastic_array_bytes: int = 0
    stochastic_arrays: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "admitted": self.admitted,
            "mesh": self.mesh,
            "cells": self.cells,
            "predicted_peak_bytes": self.predicted_bytes,
            "predicted_peak_mib": self.predicted_bytes / MIB,
            # The margin, under the name every earlier receipt used.  It is
            # no longer a flat 512 MiB: it is the model's own two named,
            # measured components, and ``margin_terms`` below says what each
            # one absorbs.
            "headroom_bytes": self.headroom_bytes,
            "margin_bytes": self.headroom_bytes,
            "required_free_bytes": self.required_bytes,
            "measured_free_bytes": self.free_bytes,
            "measured_total_bytes": self.total_bytes,
            "shortfall_bytes": self.shortfall_bytes,
            "largest_fitted_cells": self.fitted_cells,
            "fitted_registered_meshes": list(self.alternatives),
            "model_provenance": self.model_provenance,
            "measured_at_decision_time": True,
            "stochastic_basis_bytes": self.stochastic_basis_bytes,
            "stochastic_array_bytes": self.stochastic_array_bytes,
            "stochastic_arrays": dict(self.stochastic_arrays or {}),
            "stochastic_allocation_note": "logical array floor; not a measured peak; construction scratch, intermediate expressions, allocator placement and opaque library workspaces excluded",
        }
        if self.model is not None:
            record["footprint_model"] = self.model.as_dict()
            record["margin_terms"] = self.model.margin_terms()
            record["tiled_workspace_bytes"] = self.model.tiled_breakdown(self.cells)
            record["card"] = self.model.card.as_dict()
            record["configuration"] = self.model.configuration
            record["row_is_measured"] = self.model.measured
        return record


def admission_verdict(
    *,
    mesh: str,
    cells: int,
    free_bytes: int,
    total_bytes: int,
    headroom_bytes: int | None,
    registry: Mapping[str, MeshRow],
    model: ShapedFootprintModel | None = None,
    stochastic: str | None = None,
) -> AdmissionVerdict:
    """Decide whether this card, as it is right now, holds this mesh."""

    model = model or FOOTPRINT_MODEL
    margin = model.margin_bytes() if headroom_bytes is None else int(headroom_bytes)
    arrays = stochastic_array_ledger(cells, stochastic, levels=model.levels)
    basis_bytes = arrays.get("harmonic_basis", 0)
    array_bytes = sum(arrays.values())
    predicted = int(round(model.predict_bytes(cells))) + array_bytes
    # THE one admission sum, shared with the driver's floor: a byte of
    # divergence here is a card that passes one gate and dies on the other.
    required = required_free_bytes(cells, model, margin) + array_bytes
    admitted = int(free_bytes) >= required
    fitted_cells = model.max_cells(int(free_bytes), margin)
    if array_bytes:
        # The additional allocation grows with the candidate mesh too.
        low, high = 0, fitted_cells
        while low < high:
            middle = (low + high + 1) // 2
            arrays_at_middle = stochastic_array_bytes(max(4, middle), stochastic, levels=model.levels)
            if required_free_bytes(middle, model, margin) + arrays_at_middle <= int(free_bytes):
                low = middle
            else:
                high = middle - 1
        fitted_cells = low
    headroom_bytes = margin
    alternatives = tuple(
        sorted(
            row.name
            for row in registry.values()
            if row.name != mesh and row.cells <= fitted_cells
        )
    )
    return AdmissionVerdict(
        admitted=admitted,
        mesh=mesh,
        cells=int(cells),
        predicted_bytes=predicted,
        headroom_bytes=int(headroom_bytes),
        required_bytes=required,
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        shortfall_bytes=0 if admitted else required - int(free_bytes),
        fitted_cells=fitted_cells,
        alternatives=alternatives,
        model_provenance=model.provenance,
        model=model,
        stochastic_basis_bytes=basis_bytes,
        stochastic_array_bytes=array_bytes,
        stochastic_arrays=arrays,
    )


def require_admitted(verdict: AdmissionVerdict) -> None:
    """Refuse an unadmitted request, naming the fitted alternative."""

    if verdict.admitted:
        return
    model = verdict.model or FOOTPRINT_MODEL
    if verdict.headroom_bytes == model.margin_bytes():
        margin_detail = "; ".join(
            f"{name}: {_mib(term['bytes'])} for {term['absorbs']}"
            for name, term in model.margin_terms().items())
    else:
        margin_detail = f"explicit headroom {_mib(verdict.headroom_bytes)}"
    if verdict.alternatives:
        remedy = (
            "This card fits the registered mesh(es) "
            f"{', '.join(verdict.alternatives)} at this moment "
            f"({verdict.fitted_cells:,} cells fit); re-run --mesh with one of "
            "them, or free the memory the shortfall names and re-run."
        )
    else:
        core = (
            verdict.model.core_bytes
            if verdict.model is not None
            else FOOTPRINT_MODEL.core_bytes
        )
        remedy = (
            "no registered mesh fits this card at this moment: this card's "
            f"own core is {_mib(core)} before a single cell is allocated, and "
            f"this card has {_mib(verdict.free_bytes)} free.  Free device "
            "memory (close other CUDA processes and any desktop compositor "
            "holding the card) or run on a larger device."
        )
    raise _refuse(
        f"device memory admission refused --mesh {verdict.mesh}: the fitted "
        f"footprint for {verdict.cells:,} cells is "
        f"{_mib(verdict.predicted_bytes)} and the decision holds back "
        f"{_mib(verdict.headroom_bytes)}, so it needs "
        f"{_mib(verdict.required_bytes)} free; this device reports "
        f"{_mib(verdict.free_bytes)} free of {_mib(verdict.total_bytes)}, "
        f"short by {_mib(verdict.shortfall_bytes)}.  Running anyway does not "
        "fail a check -- it loads the mesh, compiles the kernels, and then "
        "dies inside a CuPy allocation part-way through the integration, "
        f"which is what this gate exists to prevent.  {remedy}  Margin: "
        f"{margin_detail}.  The row is {verdict.model_provenance}.  If "
        "this card's own ledger has been run, supply its row with "
        f"--device-fixed-mib and --device-bytes-per-cell.  {LEDGER_PROBE} is "
        "the instrument that measures it."
    )


def measure_device_memory() -> tuple[int, int]:
    """``(free, total)`` device bytes, read from the driver right now.

    Never cached and never carried between decisions: the memory another
    process is holding is part of the answer, and it changes.
    """

    for variable in ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            raise _refuse(
                f"{variable} is set, so this box has declared that no GPU work "
                "happens here and the door will not open a device to measure "
                "it.  Unset it to run a forecast on this machine, or run the "
                "forecast on the machine that owns the card."
            )
    try:
        import cupy
    except Exception as error:  # pragma: no cover - depends on the box
        raise _refuse(
            f"the CUDA lane is not importable ({error}), so free device memory "
            "cannot be measured and a forecast cannot be admitted.  Install "
            'the CUDA lane: pip install "recast-woof[gpu]", whose CUDA 12 '
            "wheel runs on a CUDA 12 or CUDA 13 driver ("
            '"woof hex[gpu-cu13]" on a CUDA 13 driver also works).  '
            "`woof hex doctor` reads your driver's CUDA major and names the "
            "extra for it."
        ) from None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise _refuse(
                "cupy imported but reports no CUDA device; there is nothing "
                "here to run a forecast on.  Check `nvidia-smi` and that "
                "CUDA_VISIBLE_DEVICES does not exclude every device."
            )
        free, total = cupy.cuda.runtime.memGetInfo()
    except ForecastDoorRefusal:
        raise
    except Exception as error:  # pragma: no cover - driver present but unusable
        if "InsufficientDriver" in str(error):
            raise _refuse(
                f"the CUDA driver refused a memory query ({error}): the "
                "installed CuPy is built for a newer CUDA major than this "
                "driver serves (cupy-cuda13x on a CUDA 12 driver).  Remove it "
                "and install the CUDA 12 wheel: pip uninstall -y cupy-cuda13x "
                '&& pip install "recast-woof[gpu-cu12]".  `woof hex doctor` '
                "reads the driver's CUDA major and names the extra for it."
            ) from None
        raise _refuse(
            f"the CUDA driver refused a memory query ({error}); the device is "
            "present but unusable, so no admission decision can be made.  "
            "Check the driver and that no other process has the card in an "
            "exclusive compute mode."
        ) from None
    return int(free), int(total)


def read_device_compute() -> tuple[int, int]:
    """The card's compute capability, read from the driver right now."""

    for variable in ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            raise _refuse(
                f"{variable} is set, so this box has declared that no GPU work "
                "happens here and the door will not open a device to read its "
                "architecture."
            )
    import cupy

    properties = cupy.cuda.runtime.getDeviceProperties(0)
    return (int(properties["major"]), int(properties["minor"]))


def read_nvrtc_targets() -> tuple[int, ...]:
    """The ``compute_XY`` targets this install's NVRTC compiles for, now."""

    from .cuda_backend.arch_admission import nvrtc_target_archs

    return nvrtc_target_archs()


def probe_numeric_route() -> dict[str, Any]:
    """Measure the numeric route on this card (one tiny compile and launch)."""

    from .cuda_backend.arch_admission import measure_numeric_route

    return dict(measure_numeric_route())


def admit_architecture() -> dict[str, Any]:
    """Answer the architecture half of "will this run?" for this card.

    Every architecture NVRTC can compile the kernels for is admitted, the
    way any other model runs on the card it is given.  The answer says
    whether the architecture holds an anchor (the numerical contract
    measured on it, :mod:`woof.hex.cuda_backend.arch_admission`) and, for an
    anchored row, the evidence pin; the receipt records the same.

    What refuses here, each naming its breakage: a card NVRTC cannot compile
    for (no kernel could be built), NVRTC that cannot be loaded at all (the
    same), and on an unanchored card, ordinary arithmetic that is not IEEE
    (every number the run produced would be wrong), which
    :func:`probe_numeric_route` measures in well under a second.  The same
    NVRTC question is the run's own gate in ``require_cuda``, so preflight's
    answer and the run's answer agree (the small-card walk once found them
    disagreeing, evidence/small-card-3080-20260824, rung 4).
    Module-level lookup of :func:`read_device_compute`,
    :func:`read_nvrtc_targets` and :func:`probe_numeric_route` is the test
    seam.
    """

    from .cuda_backend.arch_admission import (
        architecture_status,
        nvrtc_refusal,
        numeric_route_refusal,
    )

    try:
        compute = read_device_compute()
    except ForecastDoorRefusal:
        raise
    except Exception as error:  # pragma: no cover - depends on the box
        raise _refuse(
            f"the CUDA driver refused a device-properties query ({error}), so "
            "the architecture half of admission cannot be answered on this "
            "box."
        ) from None
    try:
        targets = tuple(int(value) for value in read_nvrtc_targets())
    except Exception as error:  # pragma: no cover - depends on the box
        # THE BREAKAGE: every kernel is compiled by NVRTC when the run
        # starts, so an install whose NVRTC cannot be loaded runs nothing.
        raise _refuse(
            f"NVRTC could not be asked which architectures it compiles for "
            f"({error}).  Every kernel this port launches is compiled by "
            "NVRTC when the run starts, so on this install none could be "
            "built.  `woof hex doctor` checks the CUDA wheel set."
        ) from None
    if compute[0] * 10 + compute[1] not in targets:
        # THE BREAKAGE: NVRTC cannot generate code for this card, so no
        # kernel could be built for it.
        raise _refuse(nvrtc_refusal(compute, targets))
    status = architecture_status(compute)
    record: dict[str, Any] = dict(status.as_dict())
    record["admitted"] = True
    record["nvrtc_targets"] = [f"compute_{value}" for value in targets]
    basis = status.describe()
    if not status.anchored:
        try:
            measurement = probe_numeric_route()
        except ForecastDoorRefusal:
            raise
        except Exception as error:
            # THE BREAKAGE: the probe is compiled and launched exactly the
            # way every port kernel is, so a failure here is the run's first
            # kernel failing.
            raise _refuse(
                f"a ten-line kernel compiled and launched the way every "
                f"kernel of this port is could not run on this "
                f"{status.sm} card ({type(error).__name__}: {error}); the "
                "forecast's own kernels would stop the same way."
            ) from None
        record["numeric_route"] = measurement
        refusal = numeric_route_refusal(measurement)
        if refusal is not None:
            raise _refuse(f"{status.sm}: {refusal}")
        basis += (
            "; numeric route measured on this card: FP32 subnormals "
            f"{measurement.get('ftz_route')}, guarded fallback "
            f"{measurement.get('guarded_fallback')}, a*b+c "
            f"{measurement.get('contraction')}, "
            + (
                "the same as on the anchored architectures"
                if measurement.get("matches_anchored_route")
                else "NOT the route the anchored architectures measured"
            )
        )
    record["basis"] = basis
    return record


def admit_timestep(
    mesh: str,
    dt_seconds: float,
    *,
    nominal_dx_m: float | None = None,
    convection: str = "auto",
    pbl_cadence: str = "auto",
    radiation_seconds: float = 1800.0,
    coarsest_dx_m: float | None = None,
) -> dict[str, Any]:
    """Answer the timestep half of "will this run?" from the row alone.

    RETIRED 2026-10-10 as an anchor gate (owner ruling).  This used to refuse every timestep the earned-anchor
    registry did not hold.  It now refuses only what the run cannot
    schedule -- a surface/PBL, radiation or cumulus cadence that is not a
    whole number of steps, or a timestep whose multiples are not exact in
    binary64 -- and RECORDS whether the configuration matches a registered
    anchor.  Stability is measured at bind from the mesh's own dcEdge
    (woof.hex.timestep_admission).  Card-free and file-free, so preflight's
    answer and the run's answer agree.
    """

    from . import convection_admission, dt_admission
    from . import pbl_cadence as pbl_cadence_module

    if nominal_dx_m is None:
        decision = {
            "scheme": convection_admission.SCHEME_GRELL_FREITAS,
            "constructor_scheme": "gf",
            "source": "unknown-spacing",
        }
    else:
        decision = convection_admission.convection_decision(
            nominal_dx_m=nominal_dx_m, requested=convection,
            coarsest_dx_m=coarsest_dx_m,
        )
    cumulus_scheme = decision["constructor_scheme"]
    try:
        pbl_decision = pbl_cadence_module.pbl_cadence_decision(
            dt_seconds=dt_seconds, requested=pbl_cadence,
            radiation_seconds=radiation_seconds,
        )
    except pbl_cadence_module.PblCadenceError as error:
        raise _refuse(
            f"--mesh {mesh} runs dt={float(dt_seconds):g} s and {error}"
        ) from error
    surface_pbl_seconds = float(pbl_decision["surface_pbl_seconds"])
    cumulus_seconds = None if cumulus_scheme is None else float(dt_seconds)
    try:
        steps = dt_admission.require_cadences(
            dt_seconds,
            radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme,
        )
    except dt_admission.DtAdmissionError as error:
        raise _refuse(
            f"--mesh {mesh} runs dt={float(dt_seconds):g} s with radiation "
            f"every {float(radiation_seconds):g} s: {error}.  Give --dt a "
            f"value that divides the radiation interval, or --radiation-seconds "
            f"a multiple of the timestep"
        ) from error
    return {
        "dt_seconds": float(dt_seconds),
        "admitted": True,
        "refuses_for_missing_anchor": False,
        "cadence_steps": steps,
        "anchor": dt_admission.anchor_record(
            dt_seconds,
            radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme,
        ),
        "convection": decision,
        "pbl_cadence": pbl_decision,
    }


def admit_device(
    *,
    mesh: str,
    cells: int,
    headroom_bytes: int | None,
    registry: Mapping[str, MeshRow],
    model: ShapedFootprintModel | None = None,
    stochastic: str | None = None,
) -> AdmissionVerdict:
    """Measure the card NOW, then decide.  Module-level lookup of
    :func:`measure_device_memory` is deliberate: it is the seam a test
    substitutes to exercise the decision at a free-memory value no card
    here has."""

    free, total = measure_device_memory()
    return admission_verdict(
        mesh=mesh,
        cells=cells,
        free_bytes=free,
        total_bytes=total,
        headroom_bytes=headroom_bytes,
        registry=registry,
        model=model,
        stochastic=stochastic,
    )


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastRequest:
    """One resolved, admitted-shaped forecast request."""

    repo: Path
    mesh: str
    cells: int
    dt_seconds: float
    grid: Path
    static: Path
    init: Path
    init_source: str
    start_time: str | None
    hours: float
    history_every_minutes: float
    steps: int
    capture_count: int
    out: Path
    scratch: Path
    receipt: Path
    gpuwm_checkout: Path
    case_label: str | None
    horiz_mixing: str
    convection: str
    pbl_cadence: str
    local_timestep: bool
    local_timestep_rates: tuple[int, ...]
    local_timestep_buffer_rings: int
    #: A/B instrument: an explicit per-cell rate file, ``None`` for the
    #: option's own classing from dcEdge.
    local_timestep_classing: Path | None
    #: ``None`` means "the model's own margin" -- the default.
    headroom_bytes: int | None
    #: ``None`` means "read the card and select its row at decision time",
    #: which is the default and the fix for the borrowed-row defect.  A ``_SuppliedRow``
    #: is the user's ``--device-fixed-mib``/``--device-bytes-per-cell`` pair,
    #: still waiting for the card.  A ``ShapedFootprintModel`` is a row
    #: already resolved against a card.
    model: Any
    stop_on_refusal: bool
    preflight: bool
    #: The lateral-boundary series a limited-area grid is driven by.  ``None``
    #: on a global grid; the driver refuses either mismatch by name.
    lbc_dir: Path | None = None
    history_from_minutes: float = 0.0
    fingerprint_every: int = 0
    #: The column-physics backend ROW this run selects
    #: (``woof.hex.physics_backend_admission``).  The default is the frozen
    #: lane; a provider's row runs its own pinned column batch and may
    #: declare extra prognostic scalars and a point source.
    physics_backend: str = "wsm6_column"
    #: A point-source table the selected row's seam releases from.  ``None``
    #: is no release; a row that carries no point source refuses one.
    source_table: Path | None = None
    chemistry_config: Path | None = None
    wind_farms: str | None = field(default=None, kw_only=True)
    #: Input problems collected rather than raised, preflight only.  Always
    #: empty on a real run, because there the same conditions refuse.
    input_problems: tuple[str, ...] = ()
    #: The forecast preset row (``woof.hex.forecast_preset``) this run selects.
    preset: str = forecast_preset.DEFAULT_PRESET
    #: ``"preset"`` when the row decided the surface/PBL cadence,
    #: ``"explicit"`` when --pbl-cadence was given.
    pbl_cadence_source: str = "preset"
    #: The checkpoint/restart/DA-cycling selections, forwarded verbatim to the
    #: driver, which resolves and refuses them (one decision, one source).
    cycling_argv: tuple[str, ...] = ()
    boundary_fingerprint_every: int = 0
    # MPAS-A v8.4.2 Registry defaults (woof.hex.mpas_defaults).
    time_integration_order: int = 2
    acoustic_substeps: int = 4
    radiation_seconds: float = 1800.0
    monotonic: bool = True
    #: How the run's timestep was chosen (mpas_defaults.resolve_dt): the
    #: mesh rule, the row's registered value, or an explicit --dt.
    dt_selection: Mapping[str, Any] | None = None
    #: Where the schedule knobs came from: "mpas-defaults", "explicit",
    #: "namelist" or "local-timestep" (per knob).
    schedule_sources: Mapping[str, str] | None = None
    cam_coefficient: float = 0.0
    cam_damping_levels: int = 4
    rayleigh_u: bool = False
    damping_start_m: float = 22000.0
    #: Dynamics options with no flag of their own, as (attribute, value),
    #: from ``--dynamics KEY=VALUE`` and from a ``--physics-namelist``'s
    #: dynamics groups (:func:`resolve_dynamics_overrides`).
    dynamics_overrides: tuple[tuple[str, Any], ...] = ()
    dynamics_override_sources: Mapping[str, str] | None = None
    #: ``--max-wind``: the largest horizontal wind the bind's advective
    #: stability estimate assumes; ``None`` takes the mesh row's (125 m/s).
    max_wind_m_s: float | None = None
    #: The native invariant stream the driver reads the vertical grid from,
    #: ``None`` for the init file.  :func:`resolve_invariant` decides it.
    invariant: Path | None = None
    #: How it was decided, in a sentence, for the receipt and the console.
    invariant_selection: str = "the mesh row declares no invariant stream"
    #: The named physics suite (``woof.hex.physics_suites``) and the overrides
    #: this run was resolved from; ``None`` when no suite flag was given,
    #: which is the run a bare door made before named suites.
    physics_suite: str | None = None
    physics_overrides: tuple[tuple[str, Any], ...] = ()
    stochastic: str | None = None
    #: The drag statistics a column-suite selection reads, ``None`` otherwise.
    orographic_statistics: Path | None = None
    #: The resolution's receipt, for the door's own receipt.
    physics_resolution: Mapping[str, Any] | None = None
    fullstep_local_timestep: bool = False
    fullstep_lts_mode: str = "shadow"
    # 8, not the ladder top: see the guard in drivers/fullstep_lts_forecast_v841.py
    fullstep_lts_max_rate: int = 8
    fullstep_lts_buffer_rings: int = 2
    #: Incremental analysis update (woof.hex.iau_forcing): the analysis init whose difference
    #: from ``init`` (the background) is forced in over ``iau_window_seconds``.
    iau_analysis: Path | None = None
    iau_window_seconds: float | None = None
    #: MPAS IAU increments (woof.hex.analysis_bundle, the native route's input) and the route
    #: the increment enters by (woof.hex.iau_native: "native" or "hex").
    iau_increment: Path | None = None
    iau_route: str | None = None
    #: ``--analysis``: the bundle's receipt block (woof.hex.analysis_door), None without one.
    analysis_bundle: Mapping[str, Any] | None = None
    #: Surface-pressure tendency statistics at every macro boundary.
    balance_series: bool = False
    #: ``physics.mode`` (woof.hex.correct_mode): "exact" (the matched
    #: arithmetic, the default) or "correct", and the corrections a
    #: correct-mode arm drops.  Exact hands the driver no flag at all.
    physics_mode: str = "exact"
    physics_correct_skip: tuple[str, ...] = ()

    @property
    def inputs_present(self) -> bool:
        return not self.input_problems


def default_scratch(out: Path) -> Path:
    """A scratch sibling of ``--out``: same parent, never inside the tree.

    The render door already rules that a scratch tree inside a delivered
    tree is a shipped defect; the forecast lane's kernel cache is the same
    hazard against a directory a user is about to hand to the renderer.
    """

    out = Path(out)
    return out.parent / (out.name + ".forecast-scratch")


def _require(value: Any, flag: str, breakage: str) -> Any:
    if value is None:
        raise _refuse(f"{flag} was not given and cannot be inferred: {breakage}")
    return value


def _require_file(
    path: Path, flag: str, remedy: str, collected: list[str] | None = None
) -> Path:
    """Resolve one input file.

    ``collected`` is the preflight seam.  A preflight answers "will this
    run?", and a user asking that has often not built every input yet; if a
    missing init stopped the answer, the card question -- the one no file
    fixes -- could not be asked until every file existed.  So in preflight
    the problem is RECORDED and the remaining checks still run.  In a real
    run ``collected`` is ``None`` and the same condition refuses.

    A symbolic link is followed: the file returned is its target, so the
    drivers pin, read and name in the receipt the bytes the link reaches
    (they refuse a link themselves).  Inputs found beside an input (the
    invariant stream, a deck's start-time files) are searched beside the name
    as given first, where a run directory of links keeps them.
    THE BREAKAGE THIS PREVENTS: the door refused a linked --init, so users
    passed the resolved file, and the RRFS deck's invariant.nc, staged as a
    link beside the init link, was no longer beside it; the run kept the
    init's land use and soil category (x1.2562 hrrrv5: 12,800,256 of
    29,644,818 words different by f006).
    """

    resolved = Path(path).expanduser().absolute()
    problem: str | None = None
    if not resolved.is_file():
        problem = f"{flag} names a missing file: {resolved}"
        if resolved.is_symlink():
            problem += f" (a link to {resolved.resolve()}, which does not exist)"
        problem += f".  {remedy}"
    if problem is None:
        return resolved.resolve() if resolved.is_symlink() else resolved
    if collected is None:
        raise _refuse(problem)
    collected.append(problem)
    return resolved


def _vertical_interfaces(path: Path) -> int | None:
    """``nVertLevelsP1`` of an MPAS file, ``None`` when it declares no levels."""

    from netCDF4 import Dataset

    with Dataset(path, mode="r") as dataset:
        dimensions = dataset.dimensions
        if "nVertLevelsP1" in dimensions:
            return len(dimensions["nVertLevelsP1"])
        if "nVertLevels" in dimensions:
            return len(dimensions["nVertLevels"]) + 1
    return None


def resolve_invariant(
    arguments: argparse.Namespace,
    row: MeshRow,
    grid: Path,
    static: Path,
    init: Path,
    collected: list[str] | None = None,
) -> tuple[Path | None, str]:
    """Which file the run's vertical grid comes from, and why, in a sentence.

    MPAS-A reads zgrid, zz, zb, zb3 and the rest of the metric set from the
    ``invariant`` stream when a run configures one, and from the init file
    otherwise.  A mesh row whose workflow configures the stream names its file
    (``MeshBinding.invariant_file``); this finds it beside ``--grid`` or
    ``--static`` and selects it BY DEFAULT, because fixed means default-on: a
    user who never heard of the stream gets the vertical grid the native model
    uses.  ``--invariant FILE`` names another file; ``--no-invariant`` runs on
    the init file's grid (the pre-fix behaviour, for reproducing old receipts).

    THE BREAKAGE THE DEFAULT CLOSES (measured
    2026-10-05 on RRFS v2's conus12km, physics off): the init file's grid is
    2.7 mm off in zgrid and 8e-5 relative in zz, and running on it moved hex
    from 3.6e-5 to 2.0e-4 K RMS theta against NOAA's MPAS at hour 1.
    """

    explicit = getattr(arguments, "invariant", None)
    declined = bool(getattr(arguments, "no_invariant", False))
    if explicit is not None and declined:
        raise _refuse(
            "--invariant and --no-invariant were both given.  One names the "
            "file the vertical grid comes from and the other says it comes "
            "from the init file; the door will not pick one of them for you."
        )
    if declined:
        return None, (
            "declined by --no-invariant: the vertical grid comes from the "
            "init file"
            + (f", although the {row.name} row declares {row.invariant_file}"
               if row.invariant_file else "")
        )
    if explicit is not None:
        path = _require_file(
            Path(explicit), "--invariant",
            "Name the native invariant stream file, or omit the flag.",
            collected)
        selection = "named by --invariant"
    elif row.invariant_file:
        searched: list[Path] = []
        found: Path | None = None
        # Beside the grid and static as named (a run directory of links), then
        # beside the files the links reach (see _require_file).
        named = {key: Path(value).expanduser().absolute()
                 for key in ("grid", "static")
                 if (value := getattr(arguments, key, None)) is not None}
        grid_folders = {Path(grid).parent} | (
            {named["grid"].parent} if "grid" in named else set())
        for anchor in (*named.values(), grid, static):
            if anchor is None:
                continue
            folder = Path(anchor).parent
            if folder in searched:
                continue
            searched.append(folder)
            candidate = folder / row.invariant_file
            if candidate.is_file() or candidate.is_symlink():
                found = candidate
                break
        if found is None:
            return None, (
                f"the {row.name} row declares the invariant stream "
                f"{row.invariant_file}, and no such file sits beside the grid "
                f"or static ({', '.join(str(folder) for folder in searched)}); "
                "the vertical grid comes from the init file.  On conus12km "
                "that grid moved hex by 2.0e-4 K RMS theta at hour 1 against "
                "the native model, which reads the invariant file.  Put the "
                "file beside the grid, or name it with --invariant"
            )
        path = _require_file(
            found, "--invariant",
            "The mesh row's invariant stream file must be a regular file.",
            collected)
        selection = (
            f"the {row.name} row's invariant stream, found beside the "
            f"{'grid' if found.parent in grid_folders else 'static'}"
        )
    else:
        return None, (
            "the mesh row declares no invariant stream; the vertical grid "
            "comes from the init file"
        )
    # The level count is the one property the init and the invariant must
    # share.  THE BREAKAGE THIS PREVENTS: an L65 invariant against an L60
    # init loads its 66-interface zgrid beside the init's 59-layer state and
    # dies in host preparation, after the card was reserved and the mesh
    # read, with a shape error that names neither file.
    if path.is_file() and Path(init).is_file():
        ours, theirs = _vertical_interfaces(path), _vertical_interfaces(init)
        if ours is None or ours != theirs:
            problem = (
                f"--invariant {path} carries {ours} vertical interfaces and "
                f"the init {init} carries {theirs}.  The invariant stream "
                "replaces the init file's vertical grid, so the two must "
                "share one level set; pick the invariant of the init's level "
                "file (RRFS names it <mesh>.invariant.nc_L<levels>_<model>), "
                "or pass --no-invariant"
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
    return path, selection


def _parse_rates(raw: Any, flag: str) -> tuple[int, ...]:
    try:
        rates = tuple(int(piece) for piece in str(raw).split(",") if piece.strip())
    except ValueError:
        raise _refuse(
            f"{flag} must be comma-separated integers; got {raw!r}"
        ) from None
    if not rates or rates[0] != 1 or list(rates) != sorted(set(rates)):
        raise _refuse(
            f"{flag} must be strictly increasing and start at 1; got {raw!r}.  "
            "Rate 1 is the class that carries the finest columns, and without "
            "it every column would be advanced on a longer acoustic sub-step "
            "than its own edge length admits."
        )
    return rates


def _resolve_model(arguments: argparse.Namespace) -> ShapedFootprintModel | None:
    """The user's own measured row, or ``None`` to read the card instead.

    ``None`` is the DEFAULT and it is the fix for the borrowed-row defect.  The retired
    door returned the 170 SM card's row here whenever the user did not type
    two numbers, so a 10 GiB desktop was priced with a 32 GiB card's fixed
    term and refused ``x1.40962`` and ``v15.150.38857`` -- two meshes it had
    been measured running with 2,244 MiB to spare.  The card is now read at
    decision time by :func:`resolve_admission_model`.
    """

    fixed = getattr(arguments, "device_fixed_mib", None)
    slope = getattr(arguments, "device_bytes_per_cell", None)
    if fixed is None and slope is None:
        return None
    if fixed is None or slope is None:
        raise _refuse(
            "--device-fixed-mib and --device-bytes-per-cell are one measured "
            "row and must be given together.  Half a row silently mixes this "
            "card's fixed term with another card's slope, which is a "
            "footprint nothing ever measured.  Run "
            f"{LEDGER_PROBE} on this card and pass both numbers it reports."
        )
    if float(fixed) <= 0.0 or float(slope) <= 0.0:
        raise _refuse(
            "--device-fixed-mib and --device-bytes-per-cell must both be "
            "positive; a non-positive coefficient is not a measurement of a "
            "footprint."
        )
    return _SuppliedRow(core_bytes=float(fixed) * MIB, bytes_per_cell=float(slope))


@dataclass(frozen=True)
class _SuppliedRow:
    """``--device-fixed-mib`` / ``--device-bytes-per-cell``, before the card.

    Held rather than turned into a model immediately because the model's
    tiled-workspace terms are arithmetic on the CARD, and the card is not
    read until the admission decision.
    """

    core_bytes: float
    bytes_per_cell: float

    def with_card(self, card: CardProfile, configuration: str, *,
                  rollback_armed: bool = False) -> ShapedFootprintModel:
        return ShapedFootprintModel(
            core_bytes=self.core_bytes,
            bytes_per_cell=self.bytes_per_cell,
            # A supplied row prices a bare run; arming the rollback adds its
            # measured copies on top, whoever measured the row.
            rollback_bytes_per_cell=(
                ROLLBACK_ARMED_BYTES_PER_CELL if rollback_armed else 0.0
            ),
            card=card,
            configuration=configuration,
            measured=True,
            provenance=(
                "supplied on the command line as this card's measured row; "
                "the tiled physics workspaces are charged on top of it from "
                f"this card's own {card.multiprocessors} multiprocessors, "
                "because those terms are arithmetic on the card and are not "
                "part of any measured core"
            ),
        )


def read_card_profile() -> CardProfile:
    """The card's shape, read from the driver right now.

    Two numbers -- multiprocessor count and maximum resident threads per
    multiprocessor -- and they are what every card-scaled term in the
    footprint model is priced from: the CUDA per-context local-memory
    backing store and the RRTMG column-chunk width.  Read here, never
    carried between decisions and never inferred from a card's name.
    """

    for variable in ("WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(variable, "") not in ("", "0"):
            raise _refuse(
                f"{variable} is set, so this box has declared that no GPU work "
                "happens here and the door will not open a device to read its "
                "shape.  Unset it to run a forecast on this machine."
            )
    try:
        import cupy

        properties = cupy.cuda.runtime.getDeviceProperties(0)
        threads_per_sm = cupy.cuda.runtime.deviceGetAttribute(
            cupy.cuda.runtime.cudaDevAttrMaxThreadsPerMultiProcessor, 0)
    except Exception as error:  # pragma: no cover - depends on the box
        raise _refuse(
            f"the CUDA driver refused a device-properties query ({error}), so "
            "this card's multiprocessor count cannot be read and no footprint "
            "row can be selected for it.  The footprint's largest fixed term "
            "is the per-context local-memory backing store, which CUDA prices "
            "at the card's resident-thread capacity, so a row chosen without "
            "the SM count is another card's row."
        ) from None
    # The runtime returns the device name as BYTES.  str() on bytes yields
    # "b'NVIDIA GeForce RTX 3080'", and that string goes straight into the
    # run receipt and the admission refusal -- the same defect class as a
    # refusal that prints a sentinel's repr instead of something a reader can
    # act on.  Decode it, and keep working if a driver ever returns str.
    raw_name = properties.get("name", b"")
    if isinstance(raw_name, (bytes, bytearray)):
        raw_name = raw_name.decode("utf-8", "replace")
    return card_profile_from_attributes(
        str(raw_name or "").strip() or "this device",
        {
            "MultiProcessorCount": properties["multiProcessorCount"],
            "MaxThreadsPerMultiProcessor": threads_per_sm,
        },
    )


def request_configuration(request: "ForecastRequest") -> str:
    """Which device stack this run builds -- the model's second key.

    A limited-area run builds a padded atmosphere, two levels of
    lateral-boundary state and 22 more kernels, and measures 85 %
    mesh-independent at cull cell counts.  The retired affine row had one
    configuration and over-predicted every measured cull by 1.2-1.9 GiB.
    """

    return "limited-area" if request.lbc_dir is not None else "global"


def _with_resolved_model(request: "ForecastRequest") -> "ForecastRequest":
    """The request with its footprint row pinned to the card that is here.

    Done ONCE, before the admission decision, so the verdict, the receipt and
    the ``--required-free-bytes`` the driver's own floor is handed all come
    from the same object.  Two resolutions could disagree if the card changed
    between them, and a card that passes one gate and dies on the other is
    the breakage this whole surface exists to prevent.
    """

    from dataclasses import replace

    return replace(request, model=resolve_admission_model(request))


def _pinned_for_driver(request: "ForecastRequest") -> "ForecastRequest":
    """The same-host request with a supplied row pinned to the card, once the ranks are admitted.

    Same-host admission prices every card's ranks from the supplied row itself
    (``samehost_forecast_v841.admit_request`` pins it per card), so the row
    stays unresolved until that decision is taken; the driver argv
    (``--required-free-bytes``) and the receipt are then built from the pinned
    row. Breakage it prevents: every same-host run given
    ``--device-fixed-mib``/``--device-bytes-per-cell`` was admitted and then
    stopped in build_driver_argv with "AttributeError: '_SuppliedRow' object
    has no attribute 'required_bytes'" (x1.40962, --samehost-devices 0,0,
    io-halo lane 2026-10-15). A request without a supplied row is unchanged.
    """

    if isinstance(request.model, _SuppliedRow):
        return _with_resolved_model(request)
    return request


def resolve_admission_model(request: "ForecastRequest") -> ShapedFootprintModel:
    """The footprint row this request is decided on.

    The user's supplied row if there is one, otherwise the row for the card
    the driver reports right now.  Module-level lookup of
    :func:`read_card_profile` is the seam a test substitutes to exercise the
    decision on a card this box does not have.
    """

    configuration = request_configuration(request)
    card = read_card_profile()
    # --stop-on-refusal arms the forecast rollback: two committed boundaries
    # for the late health gate and a per-step physics boundary export, which
    # a bare run does not hold (device_admission.ROLLBACK_ARMED_BYTES_PER_CELL).
    armed = bool(getattr(request, "stop_on_refusal", False))
    supplied = request.model
    if isinstance(supplied, ShapedFootprintModel):
        return supplied
    from dataclasses import replace
    interfaces = _vertical_interfaces(request.init) if getattr(request, "init", None) is not None else None
    levels = 55 if interfaces is None else interfaces - 1
    if isinstance(supplied, _SuppliedRow):
        return replace(supplied.with_card(card, configuration, rollback_armed=armed), levels=levels)
    model = model_for_card(card, configuration, rollback_armed=armed)
    if levels == model.levels:
        return model
    # The measured per-cell floor includes column state and scratch. Charge
    # every such byte at the larger column extent, in addition to the tiled
    # workspace formulas which already consume model.levels. This is a
    # conservative derived floor until this shape has its own measured row.
    scale = max(1.0, levels / model.levels)
    return replace(model, levels=levels, measured=False,
        bytes_per_cell=model.bytes_per_cell * scale,
        restated_bytes_per_cell=model.restated_bytes_per_cell * scale,
        rollback_bytes_per_cell=model.rollback_bytes_per_cell * scale,
        stage_peak_fits=(), derived_from=model.provenance,
        provenance=f"{model.provenance}; column extent {model.levels} -> {levels}, conservative derived floor")


def _schedule(hours: float, history_every_minutes: float, dt_seconds: float,
              mesh: str, history_from_minutes: float = 0.0) -> tuple[int, int]:
    """``(steps, capture_count)``, or a refusal naming the mesh's timestep.

    The timestep is the REGISTERED row's, not a module constant: the
    generated 15 km row declares 60 s where the published rows declare 120 s,
    and a schedule checked against one constant would admit a half-step run
    on the other.
    """

    total_seconds = float(hours) * 3600.0
    if total_seconds <= 0.0:
        raise _refuse(f"--hours {hours} must be positive")
    raw_steps = total_seconds / dt_seconds
    if abs(raw_steps - round(raw_steps)) > 1e-9:
        raise _refuse(
            f"--hours {hours} is not a whole number of steps on mesh {mesh}, "
            f"whose timestep for this run is {dt_seconds:.0f} s: it is "
            f"{raw_steps:.4f} steps.  A partial step cannot be integrated, so "
            f"choose a length that divides by {dt_seconds:.0f} s (for example "
            f"{max(1, int(raw_steps)) * dt_seconds / 3600.0:g} hours)."
        )
    steps = int(round(raw_steps))
    history_seconds = float(history_every_minutes) * 60
    # The stride is counted against the step itself, not its integer part: a
    # step on the eighth-second grid (18.75 s) was truncated to 18 s here, so
    # an hourly frame of a 192-step hour was given stride 200 and refused as
    # "does not divide the run" (measured 2026-10-11, 48-3 km patch mesh).
    raw_stride = history_seconds / float(dt_seconds)
    if history_seconds <= 0 or abs(raw_stride - round(raw_stride)) > 1e-9:
        raise _refuse(
            f"--history-every-minutes {history_every_minutes} is not a whole "
            f"number of {dt_seconds:g} s steps on mesh {mesh}; a history "
            "frame can only be written at a step boundary."
        )
    stride = int(round(raw_stride))
    first = float(history_from_minutes) * 60 / dt_seconds
    if first < 0 or first > steps or abs(first - round(first)) > 1e-9:
        raise _refuse('--history-from-minutes must select a whole step within the forecast')
    if (steps - int(round(first))) % stride != 0:
        raise _refuse(
            f"--history-every-minutes {history_every_minutes} does not divide "
            f"the {hours} h run on mesh {mesh} ({steps} steps, stride "
            f"{stride}); the last frame would fall inside the run rather than "
            "at its end, so the output would silently not cover the length "
            "asked for."
        )
    return steps, len(range(int(round(first)), steps + 1, stride))


def resolve_physics_backend(arguments: argparse.Namespace):
    """The backend row a request selects, or a refusal naming what exists."""

    if getattr(arguments, "chemistry_config", None) is not None:
        from .chem_transport import configure, admit_arguments
        admit_arguments(arguments)
        selected = configure(arguments.chemistry_config)
        arguments.physics_backend = selected.name
        return selected
    name = getattr(arguments, "physics_backend", None) or DEFAULT_BACKEND
    try:
        return resolve_backend(str(name))
    except ConfigurationRefusal as error:
        raise _refuse(
            f"--physics-backend {name}: {error}"
        ) from error


def backend_pin_problem(row, checkout: Path) -> str | None:
    """The seam pin the ROW is bound by, checked at the door.

    The frozen row is pinned by this port's engine manifest
    (:func:`seam_pin_problem`).  A provider's row binds a different column
    batch through its own manifest, so its adapter module publishes
    ``pin_problem(checkout)`` and answers for itself; a provider row whose
    adapter publishes none is refused here rather than checked against a
    pin it was never bound by.
    """

    if row.name == DEFAULT_BACKEND:
        return seam_pin_problem(checkout)
    try:
        module = row.load_adapter()
    except ConfigurationRefusal as error:
        return str(error)
    probe = getattr(module, "pin_problem", None)
    if probe is None:
        return (
            f"--physics-backend {row.name} names adapter module "
            f"{row.adapter_module}, which publishes no pin_problem(checkout).  "
            "That function is how a provider row says whether the named "
            "engine carries the files it executes; without it a missing file "
            "is found only after the mesh is bound, as an ImportError"
        )
    return probe(checkout)


def seam_pin_problem(checkout: Path) -> str | None:
    """What stops the frozen row's seam from running on ``checkout``, or None.

    RETIRED 2026-10-10 as a certificate gate (owner ruling; Gate law, Refusals law).  This used to refuse any engine
    whose sixteen seam files were not, byte for byte, the pinned version's:
    "the seam is the whole physics of this model ... so running against
    moved bytes would be running physics nobody proved".  That sentence
    names missing evidence, not a breakage.  MEASURED on the 8-card RTX PRO 6000 host the same day: a
    bare 15 km door preflight against woof 2.8.9 answered "pinned to woof
    2.8.7, 7 of 16 pinned files have moved" and no forecast could start.
    Physics parity was retired on 2026-08-20; a byte match with a measured
    engine is evidence the receipt records
    (:func:`woof.hex.engine_pin.measured_row_record`, ``arwen_engine_pin`` in
    the run receipt and ``engine_pin`` in the door's preflight report), never
    a permission to run.

    WHAT IS STILL REFUSED, because it cannot run: a tree that does not carry
    ``woof/core/mpas_column_batch.py``.  The seam constructs that module's
    column batch; without it Python either fails on import after the mesh is
    bound or silently imports another woof from elsewhere on the path.
    """

    from . import engine_pin

    inspection = engine_pin.inspect_seam(checkout)
    seam_module = "woof/core/mpas_column_batch.py"
    if seam_module not in inspection.absent:
        return None
    return (
        f"the woof at {checkout} carries no {seam_module}.  The physics "
        "seam constructs that module's column batch, so this tree has no "
        "physics for the run to call.  Install an engine, or point "
        f"--gpuwm-checkout at a woof source tree:\n{engine_pin.remedy()}"
    )


def seam_pin_record(checkout: Path) -> dict[str, Any]:
    """Whether ``checkout`` is a measured engine row, for the door's report.

    Evidence only; see :func:`seam_pin_problem` for why nothing refuses on it.
    """

    from . import engine_pin

    return engine_pin.measured_row_record(engine_pin.inspect_seam(checkout))


#: --convection's vocabulary -> the MPAS spelling an override records.
_CONVECTION_SPELLING = {
    "off": "off", "gf": "cu_grell_freitas", "ntiedtke": "cu_ntiedtke",
}


def _command_line_overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    """``--physics KEY=VALUE`` in order, with --convection spelled as MPAS
    spells it.  Refuses two different cumulus selections."""

    from .physics_suites import parse_override

    merged: dict[str, Any] = {}
    for text in getattr(arguments, "physics", None) or ():
        try:
            key, value = parse_override(text)
        except ConfigurationRefusal as error:
            raise _refuse(f"--physics {text}: {error}") from error
        merged[key] = value
    explicit = getattr(arguments, "convection", "auto")
    if explicit not in (None, "auto"):
        # Beyond the three hex names, --convection takes what the resolver
        # takes: an MPAS spelling, a registry option id or a cu_physics
        # number.  WOOF's registry says whether it exists.
        spelled = _CONVECTION_SPELLING.get(explicit, explicit)
        given = merged.get("config_convection_scheme", merged.get("cu_physics"))
        if given is not None and given != spelled:
            raise _refuse(
                f"--convection {explicit} and --physics "
                f"config_convection_scheme={given} name two cumulus selections "
                "for one run; give one")
        merged["config_convection_scheme"] = spelled
    return merged


def resolve_door_physics(
    arguments: argparse.Namespace, mesh: str, nominal_dx_m: float | None = None,
):
    """The named physics this request selects: ``(physics, namelist import,
    resolver overrides)``, or ``(None, None, {})`` for the legacy road.

    The legacy road is taken exactly when no physics-suite flag was given:
    then nothing about the run changes (the backend row and --convection
    decide, as before).  Otherwise the suite, the namelist and the overrides
    resolve through :func:`woof.hex.physics_suites.resolve_physics`;
    --convection, when given explicitly, is an override; --physics on top of
    a namelist is read as if the namelist had said it; and --physics-backend,
    when given, must name the row the suite resolves to: one decision, one
    source.
    """

    from .physics_namelist import read_physics_namelist
    from .physics_suites import PhysicsSelectionRefusal, generation_keys, resolve_physics

    from . import engine_registry

    suite = getattr(arguments, "physics_suite", None)
    raw = list(getattr(arguments, "physics", None) or ())
    namelist = getattr(arguments, "physics_namelist", None)
    explicit_convection = getattr(arguments, "convection", "auto")
    if (suite is None and not raw and namelist is None
            and explicit_convection in (None, "auto", *_CONVECTION_SPELLING)):
        return None, None, {}
    from .physics_default import is_arwen_default
    if is_arwen_default(suite, overrides=raw, namelist=namelist, convection=explicit_convection):
        # --physics-suite arwen_mpas_column alone: the ArWen route exactly as a bare run took it before MPAS
        # physics became the hex default (0.4.6), so the ArWen arm of every door stays byte-identical.
        return None, None, {}
    # The engine tree whose registry says which options exist: the one the
    # run names when it carries a registry (a tree that is no engine at all
    # is refused by name further down, at the pin check), else the
    # importable woof.
    engine_checkout = engine_registry.checkout_with_registry(
        getattr(arguments, "gpuwm_checkout", None))
    command_line = _command_line_overrides(arguments)
    imported = None
    if namelist is not None:
        if suite is not None:
            raise _refuse(
                f"--physics-suite {suite} and --physics-namelist {namelist} both "
                "name the run's suite; the namelist's config_physics_suite would "
                "be one of two sources for one decision.  Give one, and use "
                "--physics KEY=VALUE for overrides on top of the namelist")
        generation = {k: v for k, v in command_line.items() if k in generation_keys()}
        into_namelist = {k: v for k, v in command_line.items() if k not in generation}
        try:
            imported = read_physics_namelist(namelist, mesh=mesh, overrides=into_namelist)
        except (OSError, ConfigurationRefusal) as error:
            raise _refuse(f"--physics-namelist {namelist}: {error}") from error
        if imported.refusals:
            raise _refuse(
                f"--physics-namelist {namelist}: "
                + str(PhysicsSelectionRefusal(imported.refusals, context=str(namelist))))
        suite = imported.suite
        overrides = {**imported.overrides, **generation}
    else:
        overrides = command_line
    try:
        physics = resolve_physics(
            suite, overrides, mesh=mesh, nominal_dx_m=nominal_dx_m,
            engine_checkout=engine_checkout)
    except PhysicsSelectionRefusal as error:
        raise _refuse(f"physics suite: {error}") from error
    explicit_backend = getattr(arguments, "physics_backend", None)
    if explicit_backend not in (None, DEFAULT_BACKEND) and explicit_backend != physics.backend_row:
        raise _refuse(
            f"--physics-backend {explicit_backend} and the physics suite "
            f"({physics.suite}) resolving to row {physics.backend_row} name two "
            "backends for one run; drop --physics-backend, the suite decides it")
    return physics, imported, overrides


def resolve_schedule(arguments: argparse.Namespace, imported_namelist: Any) -> dict[str, Any]:
    """The dycore schedule: an explicit flag, else the namelist, else MPAS-A.

    MPAS-A v8.4.2's Registry defaults (woof.hex.mpas_defaults): RK order 2
    (Registry.xml:69), four acoustic sub-steps (:104), the monotonic limiter
    on (:239).  A ``--physics-namelist`` that sets ``config_dt``,
    ``config_time_integration_order``, ``config_number_of_sub_steps`` or
    ``config_monotonic`` supplies that knob when its flag is absent, so a
    namelist's dynamics run instead of being refused for disagreeing with a
    default.  ``sources`` records which of the three decided each knob.
    """

    namelist: dict[str, Any] = {}
    config = None if imported_namelist is None else imported_namelist.dycore_config
    if config is not None and imported_namelist.dycore_options:
        present = {str(option).casefold() for _g, option, _v in imported_namelist.dycore_options}
        for option, key in (
            ("config_dt", "dt"),
            ("config_time_integration_order", "order"),
            ("config_number_of_sub_steps", "substeps"),
            ("config_monotonic", "monotonic"),
        ):
            if option in present and hasattr(config, option):
                namelist[key] = getattr(config, option)
    resolved: dict[str, Any] = {"sources": {}}
    for key, flag, default in (
        ("order", "time_integration_order", mpas_defaults.DEFAULT_TIME_INTEGRATION_ORDER),
        ("substeps", "acoustic_substeps", mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS),
        ("monotonic", "monotonic", mpas_defaults.DEFAULT_MONOTONIC),
    ):
        given = getattr(arguments, flag, None)
        if given is not None:
            resolved[key], source = given, "explicit"
        elif key in namelist:
            resolved[key], source = namelist[key], "namelist"
        else:
            resolved[key], source = default, "mpas-defaults"
        resolved["sources"][key] = source
    if getattr(arguments, "dt", None) is None and "dt" in namelist:
        resolved["dt"] = float(namelist["dt"])
        resolved["sources"]["dt"] = "namelist"
    else:
        resolved["dt"] = None
        resolved["sources"]["dt"] = (
            "mpas-defaults" if getattr(arguments, "dt", None) in (None, "auto")
            else "explicit")
    resolved["order"] = int(resolved["order"])
    resolved["substeps"] = int(resolved["substeps"])
    resolved["monotonic"] = bool(resolved["monotonic"])
    return resolved


def resolve_dynamics_overrides(
    arguments: argparse.Namespace, imported_namelist: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Dynamics options with no flag of their own: ``--dynamics``, else the
    namelist.

    THE BREAKAGE THIS CLOSES (2026-10-10 audit, measured): a
    ``--physics-namelist`` with ``config_apvm_upwinding = 0.25``,
    ``config_smdiv = 0.2`` and ``config_coef_3rd_order = 1.0`` was refused
    three times with "the run has no flag for it", after the certificate
    gates that pinned those options had been retired -- the options were
    free in the configuration type and unreachable from any front door.
    Every dynamics option the namelist sets that no other flag decides now
    travels to the driver as ``--dynamics KEY=VALUE``.  An explicit
    ``--dynamics`` for an option the namelist also sets must agree with it:
    like ``--dt`` against a namelist's ``config_dt``, two different values
    for one option are refused by :func:`namelist_dycore_problems`, which
    also names an option the run's configuration does not carry.
    """

    import importlib

    driver = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    try:
        explicit = driver.parse_dynamics_overrides(getattr(arguments, "dynamics", None))
    except ValueError as error:
        raise _refuse(str(error)) from error
    overrides: dict[str, Any] = {}
    sources: dict[str, str] = {}
    config = None if imported_namelist is None else imported_namelist.dycore_config
    if config is not None and imported_namelist.dycore_options:
        from .config import _V841_DYCORE_BINDINGS

        for _group, option, _value in imported_namelist.dycore_options:
            binding = _V841_DYCORE_BINDINGS.get(str(option).casefold())
            if binding is None or binding.attribute in driver._FLAG_CARRIED_DYNAMICS:
                continue
            if not hasattr(config, binding.attribute):
                continue
            overrides[binding.attribute] = getattr(config, binding.attribute)
            sources[binding.attribute] = "namelist"
    for key, value in explicit.items():
        overrides[key] = value
        sources[key] = "explicit"
    return overrides, sources


def _physics_mode_fields(arguments: argparse.Namespace,
                         preset: "forecast_preset.ForecastPreset | None" = None) -> dict[str, Any]:
    """``--physics-mode`` / ``--physics-correct-skip`` validated at the door,
    before any input is read (woof.hex.correct_mode refuses a misspelling).

    A run that names no mode takes its preset row's (every shipped row says
    ``correct``, owner ruling 2026-10-13); ``--physics-mode exact`` is
    the matched arithmetic, byte for byte."""

    from . import correct_mode

    default = correct_mode.DEFAULT_PHYSICS_MODE if preset is None else preset.physics_mode
    try:
        skip_names = tuple(s.strip() for value in getattr(arguments, "physics_correct_skip", None) or ()
                           for s in str(value).split(",") if s.strip())
        mode, skip = correct_mode.validate(
            getattr(arguments, "physics_mode", None) or default, skip_names)
        # THE BREAKAGE THIS PREVENTS: a skip of a finding this route never applies (an MPAS-suite RP id,
        # an IAU or boundary-door id) would make an observation A/B arm identical to its control while
        # its receipt names an arm.
        foreign = [f for f in skip if f not in correct_mode.route_findings("arwen-column")]
        if foreign:
            raise correct_mode.CorrectModeRefusal(
                f"--physics-correct-skip {foreign} names findings the WOOF route never applies; it "
                f"applies {sorted(correct_mode.route_findings('arwen-column'))}")
    except correct_mode.CorrectModeRefusal as error:
        raise _refuse(str(error)) from error
    return {"physics_mode": mode, "physics_correct_skip": skip}


def default_step_selectors(physics: Any, arguments: argparse.Namespace) -> dict[str, str]:
    """The boundary-layer and cumulus selection a run's default-step cap reads.

    THE BREAKAGE THIS PREVENTS (hex 0.4.0 recut, 2026-10-16): a bare forecast names no suite, so
    ``physics`` is None and the first version of the cap read nothing and never bound -- the smoke
    forecast on x1.40962 still took the mesh's own 600 s with YSU.  With no suite named the frozen
    engine seam runs its own column suite (``forecast_preset.ENGINE_COLUMN_SUITE``, YSU), and the
    cumulus is the door's ``--convection``."""

    if physics is not None:
        return dict(physics.selectors)
    from .forecast_preset import ENGINE_COLUMN_SUITE

    convection = str(getattr(arguments, "convection", "auto") or "auto")
    return {"config_pbl_scheme": "bl_" + ENGINE_COLUMN_SUITE["boundary_layer"],
            "config_convection_scheme": "cu_grell_freitas" if convection in ("gf", "cu_grell_freitas")
            else convection}


def resolve_request(
    arguments: argparse.Namespace,
    *,
    registry: Mapping[str, MeshRow] | None = None,
) -> ForecastRequest:
    """Every check that can be made before an expensive byte is read."""
    backend_row = resolve_physics_backend(arguments)

    repo = resolve_repo(getattr(arguments, "repo", None))
    if registry is None:
        registry = load_registry(repo)

    mesh = _require(
        getattr(arguments, "mesh", None),
        "--mesh",
        "the mesh registry pins each row's grid and static bytes and its "
        "dimensions, and the door will not guess which row the supplied "
        "files are. Registered meshes: "
        + ", ".join(sorted(registry)),
    )
    if mesh not in registry:
        raise _refuse(
            f"--mesh {mesh!r} is not a registered mesh.  Registered meshes are "
            f"{', '.join(sorted(registry))}.  A row pins its grid and static "
            "by byte count and SHA-256 and declares the level and soil "
            "counts the driver sizes its arrays by; without one the door "
            "cannot size the run.  Register the row in "
            "src/hexcore/drivers/mpas_mesh_binding.py, or describe the mesh "
            "in a mesh-rows.json named by $WOOF_HEX_MESH_ROWS."
        )
    row = registry[mesh]
    preflight = bool(getattr(arguments, "preflight", False))
    collected: list[str] | None = [] if preflight else None
    physics, imported_namelist, physics_overrides = resolve_door_physics(
        arguments, mesh, getattr(row, "nominal_dx_m", None))
    if physics is not None:
        backend_row = resolve_backend(physics.backend_row)

    # ORDER.  Everything a user can fix from the command line alone is
    # checked before anything that depends on the filesystem, because those
    # are properties of the REQUEST and no file changes them.  The order
    # this replaced put --init's existence first, which meant a user with a
    # mistyped --hours and a not-yet-built init was told about the init,
    # built it, and only then learned the schedule was wrong.
    init_source = _require(
        getattr(arguments, "init_source", None),
        "--init-source",
        "the run receipt records which meteorology produced the init, and "
        "the init file's own bytes cannot say where they came from.  A "
        "forecast whose provenance sentence is blank is a forecast nobody "
        "can attribute later; state it, e.g. --init-source \"GFS 2026-08-24 "
        "00Z\"",
    )

    fingerprint_every = int(getattr(arguments, "fingerprint_every", 0))
    if fingerprint_every < 0:
        raise ForecastDoorRefusal("--fingerprint-every must be nonnegative; a negative interval cannot schedule boundary identity receipts")

    hours = float(_require(
        getattr(arguments, "hours", None), "--hours",
        "the forecast length has no default; the door will not choose how "
        "much compute a run spends"))
    history_every_minutes = float(_require(
        getattr(arguments, "history_every_minutes", None),
        "--history-every-minutes",
        "the history cadence has no default; it decides how much of the run "
        "survives it, and a wrong guess is discovered only after the run"))
    # Before the schedule is built from the row's timestep, ask whether that
    # timestep is one the frozen lane may execute at all.  Card-free,
    # file-free, and ahead of every expensive check.
    #
    # In PREFLIGHT the refusal is collected, not raised.  THE BREAKAGE THIS
    # PREVENTS (measured 2026-08-26, the proving RTX 5090): asked whether the
    # 224,210-cell row fits the card, the door exited on the timestep and
    # never printed the memory verdict at all -- so the answer to "will this
    # run?" was half an answer, which is the one thing preflight promises not
    # to be.  The two blockers are independent and a user fixing one wants to
    # know about the other in the same pass.
    convection = (
        getattr(arguments, "convection", "auto") if physics is None
        else physics.convection
    )
    radiation_seconds = getattr(arguments, "radiation_seconds", None)
    if (
        imported_namelist is not None
        and imported_namelist.radiation_seconds is not None
    ):
        namelist_seconds = imported_namelist.radiation_seconds
        if radiation_seconds is not None and float(radiation_seconds) != namelist_seconds:
            raise _refuse(
                f"--radiation-seconds {radiation_seconds:g} and the namelist's "
                f"radiation interval ({namelist_seconds:g} s) name two cadences "
                "for one run; give one")
        radiation_seconds = namelist_seconds
    if radiation_seconds is None:
        # MPAS-A config_radtlw_interval = config_radtsw_interval = 00:30:00
        # (Registry.xml:2316, 2321).
        radiation_seconds = mpas_defaults.DEFAULT_RADIATION_SECONDS
    schedule = resolve_schedule(arguments, imported_namelist)
    dynamics_overrides, dynamics_override_sources = resolve_dynamics_overrides(
        arguments, imported_namelist)
    max_wind_m_s = getattr(arguments, "max_wind", None)
    if max_wind_m_s is not None:
        max_wind_m_s = float(max_wind_m_s)
        if not (max_wind_m_s > 0.0) or max_wind_m_s != max_wind_m_s or max_wind_m_s == float("inf"):
            raise _refuse(
                f"--max-wind {max_wind_m_s!r} is not a positive finite speed "
                "in m/s, so the stability estimate has no wind to bound the "
                "step with")
    try:
        dt_selection = mpas_defaults.resolve_dt(
            getattr(arguments, "dt", None) if schedule["dt"] is None else schedule["dt"],
            nominal_dx_m=(
                None if getattr(row, "nominal_dx_m", None) is None
                else float(row.nominal_dx_m)),
            row_dt_seconds=float(row.dt_seconds),
            radiation_seconds=float(radiation_seconds),
        )
    except ValueError as error:
        raise _refuse(str(error)) from None
    if schedule["dt"] is not None:
        dt_selection["source"] = "namelist"
    if dt_selection.get("source") == "mesh":
        # YSU (and New Tiedtke) at a long step leave single hot layers at the boundary-layer top on an
        # engine without their donor bounds (woof 2.9.0 has neither); the default step takes the cap
        # woof hex global applies, and runs the mesh's own step on an engine that carries both
        # (woof.hex.global_door.default_step_cap says why and when it retires).
        from .global_door import default_step_cap
        dt_selection = default_step_cap(dt_selection, default_step_selectors(physics, arguments),
                                        radiation_seconds=float(radiation_seconds))
        if dt_selection.get("source") == "physics cap":
            print(f"woof hex: timestep {dt_selection['dt_seconds']:g} s, under the "
                  f"{dt_selection['arm']} physics arm's {dt_selection['cap_seconds']:g} s cap (the mesh's own "
                  f"step is {dt_selection['mesh_dt_seconds']:g} s): YSU at a longer step leaves single hot layers "
                  "at the boundary-layer top on an engine without its donor bounds"
                  + (f" (this engine lacks {', '.join(dt_selection['engine_lacks'])})"
                     if dt_selection.get("engine_lacks") else "")
                  + "; --dt SECONDS runs a step of your choosing", file=sys.stderr, flush=True)
    fixed_selection = dt_selection
    dt_selection = lts_step(arguments, dt_selection, float(radiation_seconds))
    # The ArWen route measured the MPAS route's loss on the same mesh (HEX-LTS-REGRESSION-2026-10-27): LTS runs
    # only where the route's measured per-step cost predicts it wins (lts_choice), at the cheapest top class.
    dt_selection = chosen_step(arguments, fixed_selection, dt_selection,
                               lts_choice(arguments, route="arwen", fixed_dt=float(fixed_selection["dt_seconds"]),
                                          lts_dt=float(dt_selection["dt_seconds"])))
    dt_selection["registered_row_dt_seconds"] = float(row.dt_seconds)
    dt_seconds = float(dt_selection["dt_seconds"])
    # The preset is a ROW (woof.hex.forecast_preset) and resolves into the
    # knob this door already has: the surface/PBL cadence at this row's
    # timestep.  An explicit --pbl-cadence is the user's own selection and
    # wins; the receipt says which of the two decided.
    try:
        preset = forecast_preset.resolve_preset(getattr(arguments, "preset", None))
    except ConfigurationRefusal as error:
        raise _refuse(str(error)) from None
    explicit_pbl_cadence = getattr(arguments, "pbl_cadence", None)
    if (
        physics is not None
        and physics.seam != "frozen"
        and preset.surface_pbl_hold_seconds is not None
        and explicit_pbl_cadence is None
    ):
        # THE BREAKAGE THIS PREVENTS: a preset row declares the frozen
        # seam's schemes (forecast_preset.ENGINE_COLUMN_SUITE) and was
        # measured and graded holding THEM; applied to a column-suite
        # selection it would hold MYNN, RUC and the GSL drag at a cadence
        # nobody measured, and the receipt would name YSU and Noah-MP.
        problem = (
            f"--preset {preset.name} holds the surface layer, land surface and "
            f"PBL up to {preset.surface_pbl_hold_seconds:g} s, measured and "
            f"graded on the frozen suite {dict(preset.suite)}; this run's "
            f"physics ({physics.suite}, row {physics.backend_row}) runs "
            f"{physics.selectors['config_pbl_scheme']} over "
            f"{physics.selectors['config_lsm_scheme']} on the {physics.seam} "
            "seam, where no hold was measured.  Use --preset "
            f"{forecast_preset.DEFAULT_PRESET}, or name the cadence with "
            "--pbl-cadence as an explicit arm")
        if collected is None:
            raise _refuse(problem)
        collected.append(problem)
    if explicit_pbl_cadence is None:
        pbl_cadence = forecast_preset.pbl_cadence_request(preset, dt_seconds)
        pbl_cadence_source = "preset"
    else:
        pbl_cadence = str(explicit_pbl_cadence)
        pbl_cadence_source = "explicit"
    if collected is None:
        admit_timestep(
            mesh, dt_seconds,
            nominal_dx_m=row.nominal_dx_m, convection=convection,
            pbl_cadence=pbl_cadence, radiation_seconds=float(radiation_seconds),
            coarsest_dx_m=getattr(row, "coarsest_dx_m", None),
        )
    else:
        try:
            admit_timestep(
                mesh, dt_seconds,
                nominal_dx_m=row.nominal_dx_m, convection=convection,
                pbl_cadence=pbl_cadence, radiation_seconds=float(radiation_seconds),
            )
        except ForecastDoorRefusal as error:
            collected.append(str(error))
    history_from_minutes = float(getattr(arguments, 'history_from_minutes', 0.0))
    steps, captures = _schedule(hours, history_every_minutes, dt_seconds, mesh, history_from_minutes)

    out = Path(_require(
        getattr(arguments, "out", None), "--out",
        "the history files and the run receipt need a destination, and the "
        "door will not write into the current directory by default")
    ).expanduser().absolute()
    if not out.parent.is_dir():
        raise _refuse(
            f"--out {out} cannot be created: its parent {out.parent} does not "
            "exist.  Create the parent first; the door refuses to build a "
            "deep path for an expensive run because a mistyped one then looks "
            "like a successful new directory."
        )
    if out.exists() and not preflight:
        raise _refuse(
            f"--out {out} exists.  A forecast writes its history frames and "
            "its receipt into a fresh directory so a second run cannot be "
            "read as the first one's continuation, or silently mix frames "
            "from two trajectories.  Give an unused path, or move the "
            "existing one aside."
        )

    scratch = Path(
        getattr(arguments, "scratch", None) or default_scratch(out)
    ).expanduser().absolute()
    if scratch == out or out in scratch.parents:
        raise _refuse(
            f"--scratch {scratch} is inside the output tree {out}.  The "
            "forecast's kernel cache is not output: leaving it inside the "
            "directory a user hands to `woof hex render` publishes "
            "half-finished temporaries as products.  Give --scratch a "
            "directory outside --out, or omit it for the sibling default."
        )
    if scratch.exists() and not preflight:
        raise _refuse(
            f"--scratch {scratch} exists.  The driver builds its kernel cache "
            "in a fresh directory so a stale cache from another engine pin "
            "cannot be loaded into this run.  Give an unused path, or remove "
            "the existing one."
        )

    horiz_mixing = getattr(arguments, "horiz_mixing", "2d_smagorinsky")
    local_timestep = bool(getattr(arguments, "local_timestep", False))
    time_integration_order, acoustic_substeps, monotonic, schedule_sources = (
        schedule["order"], schedule["substeps"], schedule["monotonic"],
        dict(schedule["sources"]))
    if local_timestep:
        # The local-timestep rate ladder is built on the RK3 six-sub-step
        # stage counts (1, 3, 6) and refuses the FCT limiter (hexcore.
        # cuda_driver_lts); with --local-timestep and no explicit choice the
        # door takes that schedule rather than refusing the flag.
        for knob, value in (("order", 3), ("substeps", 6), ("monotonic", False)):
            if schedule_sources[knob] == "mpas-defaults":
                schedule_sources[knob] = "local-timestep"
                if knob == "order":
                    time_integration_order = value
                elif knob == "substeps":
                    acoustic_substeps = value
                else:
                    monotonic = value
    rates = _parse_rates(
        getattr(arguments, "local_timestep_rates", "1,3"), "--local-timestep-rates"
    )
    rings = int(getattr(arguments, "local_timestep_buffer_rings", 1))
    if rings < 1:
        raise _refuse(
            "--local-timestep-buffer-rings must be at least 1; with no buffer "
            "ring a class boundary has no cells demoted to the finer rate and "
            "the coarse class reads values one sub-step stale."
        )
    classing_raw = getattr(arguments, "local_timestep_classing", None)
    local_timestep_classing = (
        None if classing_raw is None else Path(classing_raw).expanduser().absolute()
    )
    if local_timestep_classing is not None:
        if not local_timestep:
            raise _refuse(
                "--local-timestep-classing is an A/B instrument for the option "
                "and needs --local-timestep; without the switch the file would "
                "be read and never used, and the arm would measure nothing."
            )
        if not local_timestep_classing.is_file():
            raise _refuse(
                f"--local-timestep-classing {local_timestep_classing} is not a "
                "file. tools/lts_forced_classing.py writes one from a grid file."
            )

    # None means "the model's own margin", which is now the DEFAULT: the
    # margin is two named, measured terms priced from the card, not a flat
    # constant every card shares.  A user who names a number still gets it.
    headroom_raw = getattr(arguments, "headroom_mib", None)
    headroom_bytes: int | None
    if headroom_raw is None:
        headroom_bytes = None
    else:
        headroom_mib = float(headroom_raw)
        if headroom_mib < 0.0:
            raise _refuse("--headroom-mib must not be negative")
        headroom_bytes = int(round(headroom_mib * MIB))
    model = _resolve_model(arguments)

    # The same file checks serve the MPAS route before timestep selection.
    grid, static, init = require_forecast_inputs(arguments, collected=collected)
    invariant, invariant_selection = resolve_invariant(
        arguments, row, grid, static, init, collected)
    orographic_statistics = resolve_orographic_statistics(
        arguments, physics, mesh, grid, collected)
    for problem in microphysics_problems(backend_row, init):
        if collected is None:
            raise _refuse(problem)
        collected.append(problem)

    checkout_argument = getattr(arguments, "gpuwm_checkout", None)
    have_tree = True
    if checkout_argument is None:
        # The installed engine is the default seam tree.  All sixteen pinned
        # paths resolve from site-packages (since engine 2.5.8), and the
        # driver names an install by its version and pip's RECORD digest
        # where it names a clone by its commit, so a clone is no longer
        # required.  THE BREAKAGE THIS CLOSES: through 0.3.1 this flag was
        # mandatory, so the forecast could not run from `pip install
        # woof hex` without also cloning woof at the pinned tag.
        installed = engine_pin.installed_root()
        if installed is None:
            have_tree = False
            checkout = Path("(not installed)")
            problem = (
                "no woof is installed in this interpreter and "
                "--gpuwm-checkout was not given; the physics seam runs "
                "woof's column batch, so the forecast needs the pinned "
                "engine:\n" + engine_pin.remedy()
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        else:
            checkout = Path(installed)
    else:
        checkout = Path(checkout_argument).expanduser().absolute()
    if have_tree:
        if not checkout.is_dir():
            problem = (
                f"--gpuwm-checkout {checkout} is not a directory.  It names "
                "the woof tree whose sixteen seam files the run executes: "
                "a git clone at the pinned tag, or leave it unset to run on "
                "the installed engine."
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        else:
            problem = backend_pin_problem(backend_row, checkout)
            if problem is not None:
                if collected is None:
                    raise _refuse(problem)
                collected.append(problem)


    # The point-source table.  A row that carries no point source refuses
    # one by name: the frozen batch would accept the table and never read
    # it, which is configuration the run does not have.
    source_argument = getattr(arguments, "source_table", None)
    source_table: Path | None = None
    if source_argument is not None:
        if not backend_row.appended_scalar_names:
            problem = (
                f"--source-table {source_argument} was given, but "
                f"--physics-backend {backend_row.name} carries no point "
                "source and no appended scalars to release into; the table "
                "would be accepted and never read.  A point source selects "
                "a row that declares one: "
                f"{[name for name in registered_backend_names() if resolve_backend(name).appended_scalar_names]}"
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        source_table = _require_file(
            Path(source_argument), "--source-table",
            "the point-source table file (one waypoint per line) the "
            "treatment run releases from",
            collected,
        )

    # The lateral-boundary series.  The door checks only that the directory
    # exists and holds files: WHETHER this grid needs one is decided by the
    # grid's own bdyMask triple, in the driver, where the grid is read -- one
    # decision, one source.
    lbc_argument = getattr(arguments, "lbc_dir", None)
    lbc_dir: Path | None = None
    if lbc_argument is not None:
        lbc_dir = Path(lbc_argument).expanduser().absolute()
        if not lbc_dir.is_dir():
            problem = (
                f"--lbc-dir {lbc_dir} is not a directory.  It must be the "
                "--out-dir rw_mpas_lbc wrote its boundary files into, and it "
                "must hold lbc.*.nc: a limited-area domain integrated with no "
                "boundary series empties from its outer ring inward."
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        elif not sorted(lbc_dir.glob("lbc.*.nc")):
            problem = (
                f"--lbc-dir {lbc_dir} holds no lbc.*.nc.  rw_mpas_lbc names "
                "its output lbc.<valid-time>.nc, one file per boundary time; "
                "a directory with none of them drives nothing."
            )
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)

    if dynamics_overrides:
        # Keep what this run's configuration type carries.  A NAMELIST
        # option it does not carry is dropped here and named by the
        # comparison below when the namelist asks for something other than
        # the value the run holds; an EXPLICIT --dynamics for one is refused
        # now, because the user asked for it by name and it would not run.
        import importlib as _importlib

        _driver = _importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
        try:
            carried = _carried_overrides(
                _driver, dynamics_overrides, horiz_mixing=horiz_mixing,
                arguments=arguments, local_timestep=local_timestep) or {}
            flagged = sorted(set(carried) & _driver._FLAG_CARRIED_DYNAMICS)
            problem = None
            if flagged:
                problem = (
                    f"--dynamics sets {flagged}, and each has a flag of its "
                    f"own ({[_driver._FLAG_CARRIED_DYNAMICS_FLAGS[name] for name in flagged]}); "
                    "set it there, so one option cannot be given two values")
        except ValueError as error:
            carried, problem = {}, f"--dynamics: {error}"
        uncarried = sorted(
            key for key in dynamics_overrides
            if key not in carried and dynamics_override_sources.get(key) == "explicit")
        if problem is None and uncarried:
            problem = (
                f"--dynamics sets {uncarried}, and this run's configuration "
                "carries no such option, so the value would be recorded and "
                "not run.  The options a run carries are the fields of its "
                "configuration type (woof.hex.config_v841).")
        if problem is not None:
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        dynamics_overrides = dict(carried)
        dynamics_override_sources = {
            key: dynamics_override_sources[key] for key in dynamics_overrides}

    if imported_namelist is not None and imported_namelist.dycore_options:
        dynamics_problems = namelist_dycore_problems(
            imported_namelist, arguments,
            dt_seconds=dt_seconds, horiz_mixing=horiz_mixing,
            time_integration_order=time_integration_order,
            acoustic_substeps=acoustic_substeps, monotonic=monotonic,
            local_timestep=local_timestep, rates=rates, rings=rings,
            apply_lbcs=lbc_dir is not None,
            radiation_seconds=float(radiation_seconds),
            dynamics_overrides=dynamics_overrides,
        )
        if dynamics_problems and collected is None:
            # Every disagreement in one refusal: a user fixing the first
            # flag should not learn about the second on the next attempt.
            raise _refuse("\n".join(dynamics_problems))
        if collected is not None:
            collected.extend(dynamics_problems)

    receipt = getattr(arguments, "receipt", None)
    receipt_path = (
        Path(receipt).expanduser().absolute() if receipt is not None
        else out / "forecast-receipt.json"
    )
    if receipt is not None and not receipt_path.parent.is_dir():
        raise _refuse(
            f"--receipt directory {receipt_path.parent} does not exist; create "
            "it before the run rather than losing the receipt at the end of one."
        )

    boundary_fingerprint_every = int(getattr(arguments, "boundary_fingerprint_every", 0))
    if boundary_fingerprint_every < 0:
        raise _refuse("--boundary-fingerprint-every must be nonnegative; a negative cadence would omit the committed state evidence")
    if fingerprint_every and boundary_fingerprint_every and fingerprint_every != boundary_fingerprint_every:
        raise _refuse("the two boundary fingerprint intervals disagree; the run cannot promise two different identity schedules")
    fingerprint_every = fingerprint_every or boundary_fingerprint_every
    fullstep_enabled = bool(getattr(arguments, "fullstep_local_timestep", False))
    iau_analysis = getattr(arguments, "iau_analysis", None)
    iau_increment = getattr(arguments, "iau_increment", None)
    iau_window_seconds = getattr(arguments, "iau_window_seconds", None)
    has_increment = iau_analysis is not None or iau_increment is not None
    if has_increment != (iau_window_seconds is not None):
        raise _refuse("--iau-analysis (or --iau-increment) and --iau-window-seconds go together: an "
                      "increment without a window (or a window without an increment) has no defined forcing")
    if getattr(arguments, "iau_route", None) is not None and not has_increment:
        raise _refuse("--iau-route selects how an increment enters; this run has none (--analysis with "
                      "--iau on, --iau-analysis or --iau-increment), so it would be accepted and never used")
    mode_fields = _physics_mode_fields(arguments, preset)
    iau_route = None
    if has_increment:
        from .analysis_door import resolve_iau_route

        for flag, value in (("--iau-analysis", iau_analysis), ("--iau-increment", iau_increment)):
            if value is not None and not Path(value).expanduser().is_file():
                raise _refuse(f"{flag} {value} does not exist; the run would start from the background "
                              "with nothing to add")
        iau_analysis = None if iau_analysis is None else Path(iau_analysis).expanduser().resolve()
        iau_increment = None if iau_increment is None else Path(iau_increment).expanduser().resolve()
        if not float(iau_window_seconds) > 0:
            raise _refuse("--iau-window-seconds must be positive; a zero window is direct insertion")
        iau_route = resolve_iau_route(arguments, mode_fields["physics_mode"])
        if iau_route == "hex" and iau_analysis is None:
            raise _refuse("--iau-route hex forms its increment from two recoupled initial-conditions files "
                          "and needs --iau-analysis; an MPAS increment file alone runs --iau-route native")
    # Retired 2026-10-14 (lane W1): IAU and the balance series needed the active full-step
    # class executor, the only one with a held-tendency hook.  The fixed-step executor has the
    # same hook now (woof.hex.iau_native), and so do the same-host ranks (one class per rank);
    # only the balance series stays on one card (drivers/samehost_forecast_v841.execute_samehost).
    fullstep_knobs = {name: getattr(arguments, name, None) for name in
                     ("fullstep_lts_mode", "fullstep_lts_max_rate", "fullstep_lts_buffer_rings")}
    if not fullstep_enabled and any(value is not None for value in fullstep_knobs.values()):
        raise _refuse("the requested LTS policy would be ignored; LTS-only knobs need --fullstep-local-timestep")
    fullstep_values = {name: value for name, value in fullstep_knobs.items() if value is not None}
    if fullstep_enabled:
        from .config_fullstep_lts import validate_fullstep_lts_request
        from types import SimpleNamespace
        validate_fullstep_lts_request(SimpleNamespace(fullstep_local_timestep=True,
            **fullstep_values, local_timestep=local_timestep, lbc_dir=lbc_dir,
            time_integration_order=int(time_integration_order), acoustic_substeps=int(acoustic_substeps),
            monotonic=bool(monotonic)))
        # Retired 2026-10-11 for the active mode: rate classes are cut into
        # blocks across the cards (woof.hex.fullstep_lts_blocks), with the
        # same-class peer exchange, mailbox and register copies between cards
        # (woof.hex.cuda_fullstep_lts_multi).  The shadow keeps its refusal.
        if (getattr(arguments, "samehost_devices", None) is not None
                and fullstep_values.get("fullstep_lts_mode", "shadow") != "active"):
            raise _refuse("the full-step shadow runs on one device: the partitioned shadow has no owner-only donor reduction (halo duplicates would corrupt the rate census)")
    return ForecastRequest(
        repo=repo,
        mesh=mesh,
        cells=row.cells,
        dt_seconds=dt_seconds,
        grid=grid,
        static=static,
        init=init,
        init_source=str(init_source),
        start_time=getattr(arguments, "start_time", None),
        hours=hours,
        history_every_minutes=history_every_minutes,
        history_from_minutes=history_from_minutes,
        steps=steps,
        capture_count=captures,
        out=out,
        scratch=scratch,
        receipt=receipt_path,
        gpuwm_checkout=checkout,
        case_label=getattr(arguments, "case_label", None),
        horiz_mixing=horiz_mixing,
        time_integration_order=int(time_integration_order),
        acoustic_substeps=int(acoustic_substeps),
        radiation_seconds=float(radiation_seconds),
        monotonic=bool(monotonic),
        dt_selection=dt_selection,
        schedule_sources=schedule_sources,
        cam_coefficient=getattr(arguments,"cam_coefficient",0.0),
        cam_damping_levels=getattr(arguments,"cam_damping_levels",4),
        rayleigh_u=getattr(arguments,"rayleigh_u",False),
        damping_start_m=getattr(arguments,"damping_start_m",22000.0),
        dynamics_overrides=tuple(sorted(dynamics_overrides.items())),
        dynamics_override_sources=dict(dynamics_override_sources),
        max_wind_m_s=max_wind_m_s,
        invariant=invariant,
        invariant_selection=invariant_selection,
        convection=convection,
        pbl_cadence=pbl_cadence,
        local_timestep=local_timestep,
        local_timestep_rates=rates,
        local_timestep_buffer_rings=rings,
        local_timestep_classing=local_timestep_classing,
        headroom_bytes=headroom_bytes,
        model=model,
        stop_on_refusal=bool(getattr(arguments, "stop_on_refusal", False)),
        fingerprint_every=fingerprint_every,
        preflight=preflight,
        lbc_dir=lbc_dir,
        physics_backend=backend_row.name,
        source_table=source_table,
        chemistry_config=getattr(arguments, "chemistry_config", None),
        wind_farms=getattr(arguments, "wind_farms", None),
        input_problems=tuple(collected or ()),
        preset=preset.name,
        pbl_cadence_source=pbl_cadence_source,
        boundary_fingerprint_every=fingerprint_every,
        physics_suite=None if physics is None else physics.suite,
        physics_overrides=tuple(sorted(physics_overrides.items())),
        stochastic=resolve_stochastic_options(
            {**({} if imported_namelist is None else imported_namelist.stochastic),
             **({} if getattr(arguments, "stochastic", None) is None else
                (json.loads(arguments.stochastic) if isinstance(arguments.stochastic, str)
                 else arguments.stochastic))}).json(),
        orographic_statistics=orographic_statistics,
        physics_resolution=None if physics is None else physics.receipt(),
        iau_analysis=iau_analysis,
        iau_window_seconds=iau_window_seconds,
        iau_increment=iau_increment,
        iau_route=iau_route,
        analysis_bundle=getattr(arguments, "analysis_receipt", None),
        balance_series=bool(getattr(arguments, "balance_series", False)),
        **mode_fields,
        fullstep_local_timestep=fullstep_enabled,
        **fullstep_values,
        cycling_argv=_cycling_argv(arguments),
    )


def _preset_receipt(request: "ForecastRequest") -> dict[str, Any]:
    """The preset row as this run used it.

    A preset row declares the frozen seam's schemes.  A run whose named
    physics left that seam executed others, so its receipt does not repeat
    the row's suite as though it ran: ``suite`` is withheld and the note
    points at ``physics.resolution``, which names what did.
    """

    receipt = forecast_preset.resolve_preset(request.preset).receipt()
    resolution = request.physics_resolution
    if resolution is not None and resolution.get("seam") != "frozen":
        receipt["declared_suite"] = receipt.pop("suite")
        receipt["suite"] = None
        receipt["suite_note"] = (
            "the preset row declares the frozen seam's schemes; this run's "
            "physics left that seam, and physics.resolution names what ran")
    return receipt


def _carried_overrides(driver, overrides, *, horiz_mixing, arguments, local_timestep):
    """The NAMELIST's overrides the run's configuration type carries.

    Used only to build the comparison configuration: an option the type
    does not carry is left out here so the comparison can name it as its
    own line ("the run's configuration holds ..."), rather than failing the
    whole build on the first one.
    """

    if not overrides:
        return None
    from dataclasses import fields

    probe = driver.build_forecast_config(
        dt_seconds=60.0, horiz_mixing=horiz_mixing,
        local_timestep=bool(local_timestep),
        cam_coefficient=getattr(arguments, "cam_coefficient", 0.0),
        cam_damping_levels=getattr(arguments, "cam_damping_levels", 4),
        rayleigh_u=getattr(arguments, "rayleigh_u", False),
        damping_start_m=getattr(arguments, "damping_start_m", 22000.0),
    )
    carried = {item.name for item in fields(probe)}
    return {key: value for key, value in dict(overrides).items() if key in carried}


def namelist_dycore_problems(
    imported, arguments, *, dt_seconds, horiz_mixing, local_timestep, rates,
    rings, apply_lbcs, radiation_seconds, time_integration_order=None,
    acoustic_substeps=None, monotonic=None, dynamics_overrides=None,
) -> list[str]:
    """The namelist's dynamics options this run would not execute, as door
    problems (all of them, one pass).

    The run's configuration is the one the driver builds for this request
    (``build_forecast_config``, host only), so the comparison is against
    what executes, not against a restatement of it.  See
    :func:`woof.hex.physics_namelist.dycore_disagreements` for the measured
    breakage.
    """

    import importlib

    from .physics_namelist import dycore_disagreements

    driver = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    try:
        run_config = driver.build_forecast_config(
            dt_seconds=float(dt_seconds),
            horiz_mixing=horiz_mixing,
            local_timestep=local_timestep,
            local_timestep_rates=rates,
            local_timestep_buffer_rings=rings,
            apply_lbcs=bool(apply_lbcs),
            time_integration_order=int(
                mpas_defaults.DEFAULT_TIME_INTEGRATION_ORDER
                if time_integration_order is None else time_integration_order),
            acoustic_substeps=int(
                mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS
                if acoustic_substeps is None else acoustic_substeps),
            radiation_seconds=float(radiation_seconds),
            monotonic=bool(
                mpas_defaults.DEFAULT_MONOTONIC if monotonic is None else monotonic),
            cam_coefficient=getattr(arguments, "cam_coefficient", 0.0),
            cam_damping_levels=getattr(arguments, "cam_damping_levels", 4),
            rayleigh_u=getattr(arguments, "rayleigh_u", False),
            damping_start_m=getattr(arguments, "damping_start_m", 22000.0),
            dynamics_overrides=_carried_overrides(
                driver, dynamics_overrides, horiz_mixing=horiz_mixing,
                arguments=arguments, local_timestep=local_timestep),
        )
    except ValueError as error:
        # The driver refuses this combination by itself; the namelist
        # comparison has no configuration to hold, and says so.
        return [f"--physics-namelist: the run's dynamics configuration "
                f"cannot be built ({error}), so the namelist's dynamics "
                "options cannot be checked against it"]
    namelist = getattr(arguments, "physics_namelist", None)
    return [f"--physics-namelist {namelist}: {line}"
            for line in dycore_disagreements(imported, run_config)]


def microphysics_problems(backend_row, init) -> list[str]:
    """What stops the selected microphysics before the card, init first.

    The init check is a few reads; the table check may acquire TEMPO's pinned
    set, a download of about 930 MB into a fresh home.  THE BREAKAGE THIS
    PREVENTS (verify-hex-r2 M3, measured in a fresh HOME): the door downloaded
    the TEMPO tables and only then refused an init without aerosol fields.
    A run its init cannot start acquires nothing.
    """

    aerosol_problem = microphysics_init_problem(backend_row, init)
    if aerosol_problem is not None:
        return [aerosol_problem]
    table_problem = microphysics_table_problem(backend_row)
    return [] if table_problem is None else [table_problem]


def microphysics_init_problem(backend_row, init) -> str | None:
    """Why the init cannot start the selected microphysics, or None.

    THE BREAKAGE THIS PREVENTS: TEMPO's column seam refuses at the first
    step when the init carries no water-friendly and ice-friendly aerosol
    numbers (nwfa, nifa), after the whole construction (60 s on x1.40962,
    golden 0.4.0 R2 gate, GFS init).  A plain GFS init carries none
    (no aerosol climatology); only an init built for a TEMPO suite
    (`woof hex init --physics-suite ...`, or `woof preset prepare`, with
    woof 2.9.0's suite-aware rw_mpas_init) writes them from the WIF
    climatology.  The refusal is at the door, before the card, with the same
    reason.
    """

    if init is None:
        return None
    from .species_row import species_row_for_names
    names = (*backend_row.driven_scalar_names, *backend_row.appended_scalar_names)
    if species_row_for_names(names).engine_scheme != "tempo":
        return None
    from netCDF4 import Dataset
    import numpy as np

    try:
        dataset = Dataset(str(init))
    except OSError:
        return None  # not readable here; the init reader refuses it by name later
    with dataset:
        missing = [name for name in ("nwfa", "nifa")
                   if name not in dataset.variables
                   or not np.any(np.asarray(dataset.variables[name][:]))]
    if not missing:
        return None
    return (f"--physics-backend {backend_row.name} runs TEMPO, which needs water-friendly and "
            f"ice-friendly aerosol initial fields; {init} carries no {' or '.join(missing)} "
            "(a plain GFS-built init has no aerosol climatology). Build the init for the TEMPO "
            "suite (woof hex init --physics-suite <suite>, which writes them from the WIF "
            "climatology) or supply an init that carries nwfa and nifa")


def microphysics_table_problem(backend_row) -> str | None:
    """The Thompson coefficient tables a Thompson row reads, or a problem.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-07, conus12km): aerosol-aware Thompson reads freezeH2O.dat
    (243 MiB) and qr_acr_qg_V4.dat (71 MiB), which no recast-woof-data wheel
    carries (PyPI's per-file cap; ``woof fetch-tables`` stages them).  On a
    wheel install the run bound the mesh, prepared the host, reserved the
    card and then stopped at the first microphysics call with
    ``FileNotFoundError: missing Thompson table asset``.  The engine's own
    preflight (``woof.table_assets.require_thompson_tables``) answers the
    same question in one sentence; the door asks it before the card.
    """

    from .species_row import species_row_for_names
    names = (*backend_row.driven_scalar_names, *backend_row.appended_scalar_names)
    scheme = species_row_for_names(names).engine_scheme
    if scheme == "tempo":
        # An engine without the TEMPO port (woof before 2.8.8) has neither
        # module; refuse here, before the card, instead of an ImportError.
        try:
            from woof.tempo_assets import ensure_tempo_tables
        except ImportError as error:
            return (f"--physics-backend {backend_row.name} needs a woof engine that "
                    f"carries the TEMPO port (woof 2.8.8 or later): {error}")
        # Verify, or acquire and verify, the pinned set before the card: the
        # engine's own TEMPO entry acquires the same way.  THE BREAKAGE: this
        # check only validated, and no `woof fetch-tables` option stages the
        # TEMPO set, so a pip install was refused with a missing-file message
        # and no command that closes it (golden 0.4.0 R2, install gate).
        try:
            ensure_tempo_tables()
        except Exception as error:  # noqa: BLE001 - every acquisition failure refuses by name
            return (f"--physics-backend {backend_row.name} requires the pinned TEMPO tables and "
                    f"they could not be verified or acquired: {error}. Stage them offline with "
                    "python -c \"from woof.tempo_assets import ensure_tempo_tables; "
                    "ensure_tempo_tables(source_dir='DIR')\"")
        return None
    if scheme != "thompson_aero":
        return None
    try:
        from woof.table_assets import MissingTableAssets, require_thompson_tables
    except ImportError:
        return None  # the engine pin check names an absent or older engine
    try:
        require_thompson_tables()
    except MissingTableAssets as error:
        return f"--physics-backend {backend_row.name} runs aerosol-aware Thompson: {error}"
    return None


def resolve_orographic_statistics(arguments, physics, mesh, grid, collected):
    """The drag statistics file a column-suite selection reads.

    THE BREAKAGE THIS PREVENTS: the GSL drag reads twenty statistics per
    cell; without them it would run on zeros, which turns the drag off
    without saying so (the engine refuses that at construction, after the
    card is reserved -- the wrong place to learn it).  The default is the
    mesh's own ``<mesh>.ugwp_oro_data.nc`` beside --grid, as the native
    workflow lays it out; when that file is not there (every mesh RRFS does
    not publish one for) it is built from the terrain (woof.hex.ugwp_oro), so
    the RRFS-shape suite runs on any mesh.  A file given for a run whose drag
    reads none is refused, because it would be accepted and never read.
    """

    given = getattr(arguments, "orographic_statistics", None)
    reads = False
    spec = None
    if physics is not None and physics.column_profile is not None:
        spec = physics.column_profile.get("orographic_statistics")
        reads = spec is not None
    if not reads:
        if given is not None:
            problem = (
                f"--orographic-statistics {given} was given, but this run's "
                "physics runs no drag that reads orographic statistics; the "
                "file would be accepted and never read")
            if collected is None:
                raise _refuse(problem)
            collected.append(problem)
        return None
    if spec.get("file") == "static":
        # The statistics every MPAS static file carries (the YSU drag's
        # ten): the run's own --static is the default source.
        if given is None:
            given = getattr(arguments, "static", None)
        return _require_file(
            Path(_require(
                given, "--orographic-statistics",
                "this run's drag reads the mesh static file's var2d/con/oa/ol "
                "statistics and no --static names one")).expanduser().absolute(),
            "--orographic-statistics",
            "The engine's YSU orographic drag (WOOF gwd_opt=1) reads the ten "
            "statistics the mesh's static file carries.", collected)
    if given is None:
        # Beside --grid as named (a run directory of links) before beside the
        # file a linked grid reaches; built at the named place when neither has it.
        named = getattr(arguments, "grid", None)
        folders = list(dict.fromkeys(
            ([Path(named).expanduser().absolute().parent] if named is not None else [])
            + [Path(grid).parent]))
        given = next((folder / f"{mesh}.ugwp_oro_data.nc" for folder in folders
                      if (folder / f"{mesh}.ugwp_oro_data.nc").is_file()),
                     folders[0] / f"{mesh}.ugwp_oro_data.nc")
        if not given.is_file():
            # RRFS publishes this file for its own meshes only; every other mesh
            # gets it built from the 30" terrain the static reads (hexcore.
            # ugwp_oro), or fetched from the preparation store beside the
            # static. A preflight only reports; the real bind builds it.
            if collected is not None:
                return Path(given).expanduser().absolute()
            from .ugwp_oro import UgwpOroError, ensure
            try:
                ensure(grid, given)
            except UgwpOroError as error:
                raise _refuse(f"the GSL drag's orographic statistics for {mesh} could not be built: {error}")
    return _require_file(
        Path(given).expanduser().absolute(), "--orographic-statistics",
        "The GSL drag (config_gwdo_scheme='bl_ugwp_gwdo') reads the mesh's "
        "ugwp_oro_data.nc; RRFS v2 publishes one per mesh beside the grid "
        "(fix_rrfs2/.../meshes/<mesh>.<date>/<mesh>.ugwp_oro_data.nc).",
        collected)


def mpas_route_orographic_statistics(arguments, native_config) -> Path | None:
    """The GSL drag statistics an MPAS-route run reads when it names none: beside --grid, else built there.

    THE BREAKAGE THIS PREVENTS (hrrrv5 hex region from HRDPS, Ottawa 2026-10-07 18Z, 2026-10-29): the MPAS
    route returns before :func:`resolve_orographic_statistics`, so a run selecting ``bl_ugwp_gwdo`` (hrrrv5, the
    rrfs_* suites) on a mesh RRFS publishes no file for reached the fork's init with Registry zeros and died after
    the contract: "bl_ugwp_gwdo needs the orographic statistics of the ugwp_oro_data_in stream".  The RRFS init
    producer's receipt says the forecast door makes the file when the geography lacks GSL's topo_ugwp; on this route
    it now does, the way the column route does: ``<mesh>.ugwp_oro_data.nc`` beside --grid (RRFS's layout), built from
    the static's 30" terrain (woof.hex.ugwp_oro) when absent.  A run whose streams deck binds the stream, or whose
    drag reads no statistics, is unchanged.
    """
    options = getattr(native_config, "execution_options", None) or {}
    if str(options.get("config_gwdo_scheme", "")).strip() != "bl_ugwp_gwdo":
        return None
    namelist = getattr(arguments, "physics_namelist", None)
    if getattr(arguments, "streams", None) is not None or (
            namelist is not None and Path(namelist).with_name("streams.atmosphere").is_file()):
        return None
    grid, mesh = getattr(arguments, "grid", None), getattr(arguments, "mesh", None)
    if grid is None or mesh is None:
        return None
    from .ugwp_oro import UgwpOroError, default_path, ensure

    grid = Path(grid).expanduser().absolute()
    out = default_path(grid, mesh)
    try:
        ensure(grid, out)
    except UgwpOroError as error:
        raise _refuse(f"the GSL drag's orographic statistics for {mesh} could not be built: {error}") from None
    return out


# ---------------------------------------------------------------------------
# the driver invocation and the hand-off
# ---------------------------------------------------------------------------
def _override_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cycling_argv(arguments: Any) -> tuple[str, ...]:
    """The door's checkpoint/restart/DA selections as the driver's own flags.

    The door forwarded none of them, so a checkpoint or a DA-cycling restart
    was reachable only by calling the engineering driver past the door's mesh
    bind (which sets the run's timestep and mesh constants).
    """

    argv: list[str] = []
    hours = getattr(arguments, "checkpoint_hours", None)
    if hours:
        argv += ["--checkpoint-hours", str(hours)]
    for flag, name in (("--restart-from", "restart_from"), ("--da-analysis", "da_analysis")):
        value = getattr(arguments, name, None)
        if value is not None:
            argv += [flag, str(Path(value).expanduser().absolute())]
    if getattr(arguments, "jedi_background", False):
        argv.append("--jedi-background")
    return tuple(argv)


def _driver_floor_model(request: ForecastRequest) -> Any:
    """The row the driver's ``--required-free-bytes`` floor is priced from.

    A same-host run (``--samehost-devices``, which ``woof hex global`` always
    passes) is admitted per card by the same-host driver and never goes
    through :func:`_with_resolved_model`, so a user-supplied
    ``--device-fixed-mib``/``--device-bytes-per-cell`` row reached this argv
    unresolved and the door died with ``AttributeError: '_SuppliedRow' object
    has no attribute 'required_bytes'`` (``woof hex global --row global-30km
    --devices 0`` with both flags, 2026-10-16).  Resolve it against the card
    here, as the one-card path does before its admission decision.
    """

    if isinstance(request.model, _SuppliedRow):
        return resolve_admission_model(request)
    return request.model


def stochastic_basis_bytes(cells: int, options: str | None) -> int:
    from .stochastic import StochasticConfig
    from .device_admission import harmonic_basis_bytes
    config = StochasticConfig.parse(options)
    return harmonic_basis_bytes(cells, config.degree) if config.enabled else 0


def stochastic_array_ledger(cells: int, options: str | None, *, levels: int = 55) -> dict[str, int]:
    from .stochastic import StochasticConfig
    from .device_admission import stochastic_array_ledger as ledger
    config = StochasticConfig.parse(options)
    if not config.enabled:
        return {}
    return ledger(cells, levels, config.degree,
                  extra_scales=len(config.sppt_scales) if config.sppt else 0,
                  skeb=config.skeb)


def stochastic_array_bytes(cells: int, options: str | None, *, levels: int = 55) -> int:
    return sum(stochastic_array_ledger(cells, options, levels=levels).values())


def build_driver_argv(request: ForecastRequest) -> list[str]:
    """The argument vector handed to the engineering forecast driver.

    ``--required-free-bytes`` carries the door's own admission requirement --
    :func:`required_free_bytes` over the request's resolved model and
    headroom -- into the driver's floor, so when a card's own measured row
    replaces the shipped one at the door, the driver admits on the same
    number instead of refusing on the default model's larger fixed term.
    One number, computed once, enforced twice.
    """

    floor_model = _driver_floor_model(request)
    argv = [
        "--grid", str(request.grid),
        "--static", str(request.static),
        "--init", str(request.init),
        "--init-source", request.init_source,
        "--hours", repr(request.hours),
        "--history-every-minutes", str(request.history_every_minutes),
        "--history-from-minutes", str(request.history_from_minutes),
        "--arwen-checkout", str(request.gpuwm_checkout),
        "--horiz-mixing", request.horiz_mixing,
        "--convection", request.convection,
        "--pbl-cadence", request.pbl_cadence,
        "--required-free-bytes",
        str(required_free_bytes(request.cells, floor_model,
                                request.headroom_bytes) +
            stochastic_array_bytes(request.cells, request.stochastic,
                                   levels=floor_model.levels if floor_model is not None else 55)),
    ]
    if request.stochastic is not None:
        argv += ["--stochastic", request.stochastic]
    if request.lbc_dir is not None:
        argv += ["--lbc-dir", str(request.lbc_dir)]
    if request.invariant is not None:
        argv += ["--invariant", str(request.invariant)]
    argv += ["--time-integration-order", str(request.time_integration_order),
             "--acoustic-substeps", str(request.acoustic_substeps),
             "--radiation-seconds", repr(float(request.radiation_seconds)),
             "--monotonic" if request.monotonic else "--no-monotonic"]
    for key, value in request.dynamics_overrides:
        argv += ["--dynamics", f"{key}={json.dumps(value)}"]
    if (request.cam_coefficient,request.cam_damping_levels,request.damping_start_m)!=(0.0,4,22000.0):
        argv += ["--cam-coefficient",str(request.cam_coefficient),"--cam-damping-levels",str(request.cam_damping_levels),
                 "--damping-start-m",str(request.damping_start_m)]
    if request.rayleigh_u:argv.append("--rayleigh-u")
    if request.physics_backend != DEFAULT_BACKEND:
        argv += ["--physics-backend", request.physics_backend]
    if request.physics_suite is not None:
        argv += ["--physics-suite", request.physics_suite,
                 "--physics-mesh", request.mesh]
        for key, value in request.physics_overrides:
            argv += ["--physics", f"{key}={_override_text(value)}"]
    if request.orographic_statistics is not None:
        argv += ["--orographic-statistics", str(request.orographic_statistics)]
    if request.chemistry_config is not None:
        argv += ["--chemistry-config", str(request.chemistry_config)]
    if request.wind_farms is not None:
        argv += ["--wind-farms", request.wind_farms]
    if request.source_table is not None:
        argv += ["--source-table", str(request.source_table)]
    if request.start_time:
        argv += ["--start-time", request.start_time]
    if request.case_label:
        argv += ["--case-label", request.case_label]
    if request.local_timestep:
        argv += [
            "--local-timestep",
            "--local-timestep-rates",
            ",".join(str(rate) for rate in request.local_timestep_rates),
            "--local-timestep-buffer-rings",
            str(request.local_timestep_buffer_rings),
        ]
        if request.local_timestep_classing is not None:
            argv += ["--local-timestep-classing", str(request.local_timestep_classing)]
    if request.fingerprint_every:
        argv += ["--fingerprint-every", str(request.fingerprint_every)]
    if request.fullstep_local_timestep:
        argv += ["--fullstep-local-timestep", "--fullstep-lts-mode", request.fullstep_lts_mode,
                 "--fullstep-lts-max-rate", str(request.fullstep_lts_max_rate),
                 "--fullstep-lts-buffer-rings", str(request.fullstep_lts_buffer_rings)]
    if request.iau_window_seconds is not None:
        if request.iau_analysis is not None:
            argv += ["--iau-analysis", str(request.iau_analysis)]
        if request.iau_increment is not None:
            argv += ["--iau-increment", str(request.iau_increment)]
        argv += ["--iau-window-seconds", repr(float(request.iau_window_seconds)),
                 "--iau-route", str(request.iau_route)]
    if request.balance_series:
        argv.append("--balance-series")
    from . import correct_mode

    # Exact adds nothing: the driver sees the argument vector it always did.
    argv += correct_mode.door_argv(request.physics_mode, request.physics_correct_skip)
    argv += list(getattr(request, "cycling_argv", ()))
    if request.stop_on_refusal:
        argv.append("--stop-on-refusal")
    if request.preflight:
        argv.append("--preflight-only")
    else:
        argv += [
            "--cache-root", str(request.scratch),
            "--output", str(request.out),
        ]
    return argv


def render_command(request: ForecastRequest, history: Sequence[Path]) -> list[str] | None:
    """The exact ``woof hex render`` command that consumes this output.

    The forecast door's job ends at files the NEXT door can take.  Printing
    the command is what makes that true for a reader rather than only for a
    manual: the converter needs the run's own grid file and the run's start
    time to number lead hours, and both are known here and nowhere else.
    """

    if not history:
        return None
    command = ["woof hex", "render", "--history"]
    command += [str(path) for path in history]
    command += [
        "--mesh", str(request.grid),
        "--out", str(request.out / "png"),
    ]
    if request.start_time:
        command += ["--simulation-start", request.start_time]
    return command


def products_command(request: ForecastRequest, history: Sequence[Path],
                     labels: Sequence[str] = ()) -> list[str]:
    """The ``woof hex products`` command for this run's output; ``labels``
    are the door's forwarded ``--model-label``/``--brand`` flags."""

    command = ["woof hex", "products", "--forecast", str(request.out),
               "--mesh", str(request.grid), "--init", str(request.init),
               "--out", str(request.out / "products")]
    if request.start_time:
        command += ["--simulation-start", request.start_time]
    return [*command, *labels]


def release_device_memory() -> dict[str, Any] | None:
    """Hand the forecast's device memory back before the product stage.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-12): the product stage is
    CPU work in Rust binaries, and it ran inside the forecast process while
    that process still held every byte the integration had allocated -- on
    a 30 km global run about 84 GB of one card for the 26 minutes the maps
    took, memory no other job on the box could use.  The pools are emptied
    here; what the driver still references stays, and the record says how
    much, so "released" is a number and not a hope.  Returns ``None`` when
    there is no CUDA runtime to ask (a CPU proof run)."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        # No device visible (the card was given back, woof.hex.card_lease): nothing is held, and asking would only
        # fail; never open a context here.
        return None
    try:
        import gc

        import cupy
    except Exception:  # noqa: BLE001 - no device stack, nothing held
        return None
    try:
        free_before, total = cupy.cuda.runtime.memGetInfo()
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        free_after, _ = cupy.cuda.runtime.memGetInfo()
    except Exception as error:  # noqa: BLE001 - never fail a finished forecast
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "device_total_bytes": int(total),
        "device_free_before_bytes": int(free_before),
        "device_free_after_bytes": int(free_after),
        "released_bytes": int(free_after - free_before),
        "still_in_use_bytes": int(total - free_after),
    }


def products_command_for_this_install(request: ForecastRequest,
                                      history: Sequence[Path],
                                      labels: Sequence[str] = ()) -> list[str]:
    """The default product stage's command, without point pictures when this
    install has no ``rw_hex_points``.

    THE BREAKAGE THIS PREVENTS: woof 2.9.0 builds ``rw_hex_points`` as a
    standalone program, not a wheel tool, and the products door refuses
    before any view when it is absent (so a missing point tool is not found
    after every map).  As the forecast's automatic last stage that refusal
    left a pip install with no maps at all, though maps never need the point
    tool.  Here the automatic stage draws the maps and names what it skipped
    and how to get it; ``woof hex products`` run by hand keeps its refusal.
    A missing converter or renderer is left to the door's own refusal."""

    from .products_door import ProductDoorError, resolve_points_exe
    from . import render_door

    command = products_command(request, history, labels)
    try:
        convert_exe = render_door.resolve_convert_exe(None)
    except MpasPortError:
        return command
    try:
        resolve_points_exe(None, convert_exe)
    except ProductDoorError as error:
        reason = str(error).splitlines()[0]
        print(f"PRODUCTS soundings and meteograms skipped: {reason}; the maps "
              "are drawn. Install rw_hex_points and rerun: "
              + " ".join(command), flush=True)
        return [*command, "--no-points"]
    return command


def start_detached_products(request: ForecastRequest,
                            history: Sequence[Path],
                            command: list[str] | None = None) -> int:
    """Start the product door as its own process and return at once.

    The forecast process then exits and its cards are free while the maps
    draw on CPU: what a shared box wants, where a card held through the
    product stage is a card another job is waiting for.  The child is given
    no CUDA device, leads its own session (so the forecast's exit, or the
    queue wrapper's, does not take it down) and writes ``products.log``
    beside the frames."""

    import subprocess

    command = command or products_command(request, history)
    log_path = request.out / "products.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    with log_path.open("ab") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "woof.hex.cli", *command[1:]],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True)
    print(f"PRODUCTS detached pid={child.pid} log={log_path} "
          f"out={request.out / 'products'}", flush=True)
    print("PRODUCTS $ " + " ".join(command), flush=True)
    return 0


def run_default_products(request: ForecastRequest, history: Sequence[Path],
                         arguments: argparse.Namespace) -> int:
    """Draw the standard product set into ``<out>/products``; returns the
    product door's exit code (2 for a named refusal)."""

    from .products_door import add_products_arguments, label_argv, run_products

    command = products_command_for_this_install(request, history, label_argv(arguments))
    # Detached is the default.  THE BREAKAGE IT PREVENTS (hex, a rented box,
    # 2026-10-13): the 20 -> 2.5 km global forecast finished at 02:13Z on
    # four same-host cards and its in-command product stage (CPU-only Rust)
    # then held ~66 GB on each of the four cards for 30 minutes, with the
    # next two queued 4-card legs blocked behind it.  Emptying the pools
    # (release_device_memory) cannot help there: it frees only this thread's
    # pool on one device, while the other ranks' pools and every card's CUDA
    # context live until the process exits.  So the process exits.
    if getattr(arguments, "products_detached", True):
        return start_detached_products(request, history, command)
    released = release_device_memory()
    if released is not None:
        if "error" in released:
            print(f"DEVICE memory not released before products: {released['error']}",
                  flush=True)
        else:
            print(
                f"DEVICE released {released['released_bytes'] / 2**20:.0f} MiB before "
                f"products; {released['still_in_use_bytes'] / 2**20:.0f} MiB of "
                f"{released['device_total_bytes'] / 2**20:.0f} MiB still in use on the "
                f"card (all tenants)", flush=True)
    print("PRODUCTS $ " + " ".join(command), flush=True)
    parser = argparse.ArgumentParser(prog="woof hex products")
    add_products_arguments(parser)
    product_arguments = parser.parse_args(command[2:])
    try:
        code = int(run_products(product_arguments))
    except (MpasPortError, OSError, ValueError) as error:
        print(f"PRODUCTS refused: {error}", flush=True)
        print("NEXT " + " ".join(command), flush=True)
        return 2
    if code != 0:
        print("PRODUCTS incomplete; rerun: " + " ".join(command), flush=True)
    return code


def history_files(out: Path) -> list[Path]:
    """The history frames the driver wrote, in valid-time order.

    The WOOF route writes ``cuda-history.*.nc``.  The MPAS suite route writes MPAS's own stream files
    (``history.*`` or the template its request names), found the way ``native_default_products`` finds them.
    THE BREAKAGE THIS PREVENTS (hex parity run group 7, #27): ``pair --physics-suite ...`` reported
    "frames control 0, treatment 0" beside legs that had each written their frames.
    """

    out = Path(out)
    frames = sorted(out.glob("cuda-history.*.nc"))
    if frames or not (out / "native-physics-request.json").is_file():
        return frames
    from .products_door import native_output_frames
    return (sorted(out.glob("history.*.nc")) or sorted(out.glob("output.*.nc"))
            or native_output_frames(out))


def native_default_products(arguments: argparse.Namespace) -> None:
    """Finish the MPAS suite door with the same default products stage."""
    if getattr(arguments, "preflight", False) or getattr(arguments, "no_products", False):
        return
    from types import SimpleNamespace
    out = Path(arguments.out).resolve()
    from .products_door import native_output_frames
    history = (sorted(out.glob("history.*.nc")) or sorted(out.glob("output.*.nc"))
               or native_output_frames(out))
    if not history:
        return
    plan = json.loads((out / "native-physics-request.json").read_text())
    start = plan["start"]
    from .fullstep_lts_admission import moving_launch_switches
    if moving_launch_switches() and getattr(arguments, "start_time", None):
        # A moving segment starts at its carrier's time, but its maps belong to the run: the ArWen route labels
        # them with the run's init (the door's --start-time) and leads from it.  THE BREAKAGE THIS PREVENTS
        # (HEX-PHYSICS-PARITY run group 2 #5/#6): after a move every MPAS-route map read "Init 10/24 01Z | +000:30"
        # (the segment start) and filed under mn...-2026102401 instead of the run's 00Z init and +1:30 lead.
        start = arguments.start_time
    request = SimpleNamespace(out=out, grid=Path(plan["grid"] or plan["init"]),
                              init=Path(plan["init"]), start_time=start)
    run_default_products(request, history, arguments)


def _correct_mode_record(request: "ForecastRequest") -> dict[str, Any]:
    from . import correct_mode

    return correct_mode.run_record(request.physics_mode, request.physics_correct_skip)


def build_receipt(
    *,
    request: ForecastRequest,
    admission: AdmissionVerdict | None,
    bind_receipt: Mapping[str, Any] | None,
    driver_receipt: Mapping[str, Any] | None,
    history: Sequence[Path],
    driver_argv: Sequence[str],
    seconds: float,
    status: str,
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The door's own receipt: what ran, on what, and what comes next.

    ``architecture`` is :func:`admit_architecture`'s record: the compute
    capability, whether it holds an anchor and which evidence pin, and on an
    unanchored card the numeric route measured there.  ``None`` only when
    the door stopped before asking.
    """

    from . import __version__

    return {
        "schema": RECEIPT_SCHEMA,
        "tool": "woof hex forecast",
        "door_version": __version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "host": host_record(),
        "status": status,
        "mesh": {
            "name": request.mesh,
            "cells": request.cells,
            "dt_seconds": request.dt_seconds,
            "dt_selection": (
                None if request.dt_selection is None else dict(request.dt_selection)
            ),
            "grid": str(request.grid),
            "static": str(request.static),
        },
        "init": {"path": str(request.init), "source": request.init_source},
        "vertical_grid": {
            "invariant": None if request.invariant is None else str(request.invariant),
            "selection": request.invariant_selection,
        },
        "physics": {
            "backend": request.physics_backend,
            "suite": request.physics_suite,
            "resolution": (
                None if request.physics_resolution is None
                else dict(request.physics_resolution)
            ),
            "orographic_statistics": (
                None if request.orographic_statistics is None
                else str(request.orographic_statistics)
            ),
            "source_table": (
                None if request.source_table is None
                else str(request.source_table)
            ),
            # The run's own surface/PBL call count against the cadence it
            # declared (woof.hex.drivers.run_cuda_v841_forecast._physics_cadence_proof):
            # a held cadence that reproduced the welded arm exactly would be
            # first evidence the hold never happened, and this is where a
            # reader checks.  ``None`` until the driver has run.
            "cadence": (
                None
                if driver_receipt is None
                else dict(driver_receipt).get("forecast", {}).get(
                    "physics_cadence"
                )
            ),
        },
        "schedule": {
            "hours": request.hours,
            "steps": request.steps,
            "history_every_minutes": request.history_every_minutes,
            "history_from_minutes": request.history_from_minutes,
            "expected_frames": request.capture_count,
            "start_time": request.start_time,
        },
        "configuration": {
            "horiz_mixing": request.horiz_mixing,
            "time_integration_order": request.time_integration_order,
            "acoustic_substeps": request.acoustic_substeps,
            "dynamics_split_steps": mpas_defaults.DEFAULT_DYNAMICS_SPLITS,
            "radiation_seconds": request.radiation_seconds,
            "monotonic": request.monotonic,
            "schedule_sources": (
                None if request.schedule_sources is None
                else dict(request.schedule_sources)
            ),
            "mpas_registry_defaults": {
                knob: dict(row) for knob, row in mpas_defaults.REGISTRY_DEFAULTS.items()
            },
            "cam_coefficient":request.cam_coefficient,"cam_damping_levels":request.cam_damping_levels,
            "rayleigh_u":request.rayleigh_u,"damping_start_m":request.damping_start_m,
            "max_wind_m_s": request.max_wind_m_s,
            "dynamics_overrides": dict(request.dynamics_overrides),
            "dynamics_override_sources": (
                None if request.dynamics_override_sources is None
                else dict(request.dynamics_override_sources)
            ),
            "convection": request.convection,
            "preset": _preset_receipt(request),
            "pbl_cadence": request.pbl_cadence,
            "pbl_cadence_source": request.pbl_cadence_source,
            "local_timestep": request.local_timestep,
            "local_timestep_rates": list(request.local_timestep_rates),
            "local_timestep_buffer_rings": request.local_timestep_buffer_rings,
            "local_timestep_classing": (
                None
                if request.local_timestep_classing is None
                else str(request.local_timestep_classing)
            ),
            "stop_on_refusal": request.stop_on_refusal,
            "fingerprint_every": request.fingerprint_every,
            "iau": (None if request.iau_window_seconds is None else
                    {"analysis": None if request.iau_analysis is None else str(request.iau_analysis),
                     "increment": None if request.iau_increment is None else str(request.iau_increment),
                     "route": request.iau_route,
                     "window_seconds": request.iau_window_seconds}),
            "balance_series": request.balance_series,
            # physics.mode, recorded only when a run asked for correct mode,
            # so an exact receipt is the one written before the key existed.
            **(
                {"physics_mode": {"physics.mode": request.physics_mode,
                                  "physics.correct_skip": list(request.physics_correct_skip),
                                  # what the run's engine resolutions applied and every named
                                  # gap (woof.hex.correct_mode): never a refusal, always here
                                  "correct_mode": _correct_mode_record(request)}}
                if request.physics_mode != "exact" else {}
            ),
        },
        **({"analysis_bundle": dict(request.analysis_bundle)}
           if request.analysis_bundle is not None else {}),
        "admission": admission.as_dict() if admission is not None else None,
        # kernel numerics this run compiled with: fast (the default) or strict (bitwise to native MPAS)
        "numerics": _hex_numerics_receipt(),
        "architecture": None if architecture is None else dict(architecture),
        # ``None`` when the card could not be read at all, which is the one
        # case where the door has no row to report rather than the wrong one.
        "footprint_model": (
            request.model.as_dict()
            if isinstance(request.model, ShapedFootprintModel)
            else None
        ),
        "mesh_binding": bind_receipt,
        "driver_receipt": driver_receipt,
        "driver_argv": list(driver_argv),
        "gpuwm_checkout": str(request.gpuwm_checkout),
        "repo": str(request.repo),
        "out": str(request.out),
        "scratch": str(request.scratch),
        "history": [str(path) for path in history],
        "render_command": render_command(request, history),
        "door_seconds": round(float(seconds), 3),
        **({"fullstep_lts": {"mode": request.fullstep_lts_mode,
                            "max_rate": request.fullstep_lts_max_rate,
                            "buffer_rings": request.fullstep_lts_buffer_rings,
                            "admission_complete": False}}
           if request.fullstep_local_timestep else {}),
    }


# ---------------------------------------------------------------------------
# the handler
# ---------------------------------------------------------------------------
def _write_receipt(request: ForecastRequest, receipt: Mapping[str, Any]) -> Path | None:
    path = request.receipt
    if not path.parent.is_dir():
        return None
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _load_drivers(drivers: Path):
    """The two driver modules, loaded from the package's drivers directory.

    The mesh binding is re-executed per run on purpose: a candidate-timestep
    override (:mod:`woof.hex.mesh_row_candidate`) is applied where that module
    builds its registry, so a fresh execution is how a run sees it.
    """

    binding = _load_module("mpas_mesh_binding", Path(drivers) / "mpas_mesh_binding.py")
    driver = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    return binding, driver


_BIND_REMEDY = (
    "  The registry pins each row's grid and static by byte count and "
    "SHA-256 and admits its timestep against the file's own dcEdge, so this "
    "is the supplied files disagreeing with the named row, not a transient "
    "failure.  `woof hex mesh-check --grid ... --static ...` prints the "
    "digests the files actually carry."
)


def _bind(binding, driver, request: ForecastRequest) -> dict[str, Any]:
    # The whole schedule and the run's timestep, always: the bind measures
    # stability and cadence integrality against what the run executes.
    options = dict(
        time_integration_order=request.time_integration_order,
        acoustic_substeps=request.acoustic_substeps,
        radiation_seconds=request.radiation_seconds,
        dt_seconds=request.dt_seconds,
        max_wind_m_s=request.max_wind_m_s,
        apvm_upwinding=float(dict(request.dynamics_overrides).get("config_apvm_upwinding", 0.5)),
    )
    try:
        return binding.bind_mesh(
            driver.proof,
            request.mesh,
            grid=request.grid,
            static=request.static,
            forecast=driver,
            convection=request.convection,
            pbl_cadence=request.pbl_cadence,
            **options,
        )
    except binding.MeshBindingError as error:
        # The file-identity remedy belongs to a file-identity refusal only; a
        # stability or cadence refusal names its own remedy (a smaller --dt,
        # more sub-steps, a cadence the step divides).
        text = str(error)
        physical = ("unstable mesh/timestep" in text
                    or "not a positive integer multiple" in text
                    or "clock closure" in text)
        raise _refuse(f"the mesh bind refused: {text}"
                      f"{'' if physical else _BIND_REMEDY}") from None


def _print_admission(request: ForecastRequest, admission: AdmissionVerdict) -> None:
    model = admission.model
    card = (
        f"{model.card.multiprocessors} SM"
        if model is not None
        else "unknown card"
    )
    row = (
        ("measured" if model.measured else "DERIVED")
        + f" {model.configuration}"
        if model is not None
        else "default"
    )
    print(
        "ADMISSION mesh={mesh} cells={cells:,} card={card} row={row} "
        "predicted={predicted} margin={margin} free={free} of {total} "
        "-> {verdict}".format(
            mesh=request.mesh,
            cells=request.cells,
            card=card,
            row=row,
            predicted=_mib(admission.predicted_bytes),
            margin=_mib(admission.headroom_bytes),
            free=_mib(admission.free_bytes),
            total=_mib(admission.total_bytes),
            verdict="admitted" if admission.admitted else "REFUSED",
        ),
        flush=True,
    )
    print(f"VERTICAL GRID {request.invariant_selection}"
          + ("" if request.invariant is None else f": {request.invariant}"),
          flush=True)


def _run_preflight(request: ForecastRequest, registry: Mapping[str, MeshRow],
                   started: float) -> int:
    """Answer the whole question, not the first half of it.

    A preflight that stopped at the first blocker would send a user round the
    loop once per blocker, on a lane where one loop is a mesh bind and a
    device query.  So every check runs, every result is reported, and the
    EXIT CODE carries the verdict.  This is the one mode where an
    unadmitted request is reported rather than raised.
    """

    problems: list[str] = list(request.input_problems)
    for problem in request.input_problems:
        print(f"INPUT MISSING {problem}", flush=True)

    bind_receipt: dict[str, Any] | None = None
    if request.grid.is_file() and request.static.is_file():
        binding, driver = _load_drivers(request.repo)
        try:
            bind_receipt = _bind(binding, driver, request)
            print(
                f"BIND mesh={request.mesh} rebound={bind_receipt['rebound']} "
                f"dt={bind_receipt['timestep_admission']['requested_dt_seconds']} s",
                flush=True,
            )
        except ForecastDoorRefusal as error:
            problems.append(str(error))
            print(f"BIND REFUSED {error}", flush=True)
    else:
        driver = None
        print(
            "BIND NOT ATTEMPTED the mesh pair is not both present, and the "
            "bind is a cross-examination of those exact bytes",
            flush=True,
        )

    admission: AdmissionVerdict | None = None
    try:
        request = _with_resolved_model(request)
        admission = admit_device(
            mesh=request.mesh, cells=request.cells,
            headroom_bytes=request.headroom_bytes, registry=registry,
            model=request.model,
            stochastic=request.stochastic,
        )
        _print_admission(request, admission)
        if not admission.admitted:
            try:
                require_admitted(admission)
            except ForecastDoorRefusal as error:
                problems.append(str(error))
    except ForecastDoorRefusal as error:
        problems.append(str(error))
        print(f"ADMISSION NOT MEASURED {error}", flush=True)

    architecture: dict[str, Any] | None = None
    try:
        architecture = admit_architecture()
        print(
            f"ARCHITECTURE sm={architecture['sm']} -> admitted, "
            f"{architecture['basis']}",
            flush=True,
        )
    except ForecastDoorRefusal as error:
        problems.append(str(error))
        print(f"ARCHITECTURE REFUSED {error}", flush=True)

    driver_receipt: dict[str, Any] | None = None
    driver_argv: list[str] = []
    if not problems and driver is not None:
        driver_argv = build_driver_argv(request)
        try:
            rc = int(driver.main(driver_argv))
        except MpasPortError as error:
            # A refusal from this program names the breakage it prevents (the
            # gate law), so it is relayed verbatim: wrapping it in "stopped
            # with <ExceptionName>" buries the sentence a user needs behind a
            # Python type they cannot act on.
            problems.append(str(error))
            rc = 1
        except Exception as error:
            # An exception that is NOT one of this program's named refusals
            # reached the door.  Relaying it alone was a gate-law defect in
            # its own right and it had a measured victim: before 2026-08-26 a
            # limited-area mesh handed to this door reported
            # "RuntimeError: cellsOnEdge must contain two valid one-based
            # endpoints" -- an array-index sentence -- when what was actually
            # true is that the driver had no lateral-boundary route at all.
            # Say both: what stopped, and that its silence about the reason
            # is a defect rather than the user's error.
            problems.append(
                f"the driver's preflight stopped with "
                f"{type(error).__name__}: {error}.  That is not one of this "
                f"program's named refusals, so it does not say which "
                f"configuration it is refusing or why; treat it as a defect "
                f"in the driver rather than a statement about this run."
            )
            rc = 1
        if rc != 0:
            problems.append(f"the driver's preflight exited {rc}")

    status = "preflight_passed" if not problems else "preflight_refused"
    receipt = build_receipt(
        request=request, admission=admission, bind_receipt=bind_receipt,
        driver_receipt=driver_receipt, history=[], driver_argv=driver_argv,
        seconds=time.monotonic() - started, status=status,
        architecture=architecture,
    )
    receipt["preflight_problems"] = problems
    written = _write_receipt(request, receipt)
    if written is not None:
        print(f"RECEIPT {written}", flush=True)
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    for problem in problems:
        print(f"PREFLIGHT REFUSED {problem}", file=sys.stderr)
    print(
        f"PREFLIGHT mesh={request.mesh} problems={len(problems)} status={status}",
        flush=True,
    )
    return 1 if problems else 0


#: The mode an MPAS suite route takes at this door when the run names none (``--physics-mode`` or the
#: namelist's config_physics_mode).  The library default in ``mpasphys.driver_v842.resolve_namelist`` stays
#: exact, so replay and reproduction callers are unchanged.
PHYSICS_MODE_DEFAULT = "correct"


def announce_physics_mode(config: Any, arguments: argparse.Namespace) -> None:
    """In correct mode, print the mode and a loud multi-line warning naming every gap at startup
    (it begins ``CORRECT MODE: named gaps``).

    Exact mode prints nothing, so an exact run's console is the one written before the switch existed."""
    if config.mode == "exact":
        return
    source = ("--physics-mode" if getattr(arguments, "physics_mode", None) is not None
              else "namelist config_physics_mode or the door default for an MPAS suite route")
    print(f"PHYSICS MODE {config.mode} ({source}); corrections applied: "
          f"{', '.join(sorted(config.corrections)) or 'none'}; --physics-mode exact runs native "
          "MPAS v8.4.2 bit for bit", file=sys.stderr, flush=True)
    warning = config.gap_warning()
    if warning:
        print(warning, file=sys.stderr, flush=True)


def _hex_numerics_receipt() -> dict[str, Any]:
    from . import numerics
    return numerics.receipt_block()


def announce_cpu_slots(config: Any) -> None:
    """Print, at start-up, every active MPAS-suite slot that runs on the CPU in this run (lines beginning
    ``PHYSICS ON CPU:``; woof.hex.mpasphys.driver_v842.CpuSlot).  A run with every active slot on the GPU
    prints nothing."""
    warning = getattr(config, "cpu_warning", lambda: None)()
    if warning:
        print(warning, file=sys.stderr, flush=True)


def mpas_suite_lts_refusal(arguments: argparse.Namespace, environ=None) -> str | None:
    """The refusal for a local-time-stepping switch on the MPAS suite route, or None.

    THE BREAKAGE THIS PREVENTS: the MPAS suite route returns to its own driver before any LTS flag is
    read, so ``--physics-suite X --fullstep-local-timestep`` (or ``--local-timestep``) ran the fixed step
    and accepted the flag silently.  A moving segment is exempt: the moving controller passes its own fixed
    LTS flags to every segment, and the moving-route admission judges a moving region instead.
    """

    from .fullstep_lts_admission import moving_launch_switches

    if moving_launch_switches(environ):
        return None
    if getattr(arguments, "local_timestep", False):
        # THE BREAKAGE THIS PREVENTS: --local-timestep is the acoustic rate ladder (woof.hex.cuda_driver_lts),
        # built on the RK3 six-sub-step schedule and refusing the FCT limiter; the MPAS suite's own transport is
        # monotonic FCT (config_monotonic), so the ladder cannot step this configuration.  Full-step LTS can.
        return ("--local-timestep is the acoustic rate ladder, which needs RK3 with six acoustic sub-steps and no "
                "FCT limiter; the MPAS suite route steps MPAS-A's RK2 schedule with monotonic FCT transport.  Run "
                "full-step local time stepping on this route instead: --fullstep-local-timestep "
                "--fullstep-lts-mode active (or the shadow observer without --fullstep-lts-mode)")
    knobs = {name: getattr(arguments, name, None) for name in
             ("fullstep_lts_mode", "fullstep_lts_max_rate", "fullstep_lts_buffer_rings")}
    if not getattr(arguments, "fullstep_local_timestep", False):
        if any(value is not None for value in knobs.values()):
            return "the requested LTS policy would be ignored; LTS-only knobs need --fullstep-local-timestep"
        return None
    # Retired (HEX-PHYSICS-PARITY run group 2 #8; owner ruling 2026-10-24: every hex capability works on every physics
    # route): full-step LTS runs on this route, active (woof.hex.mpasphys.lts_v842, every rate class with the MPAS
    # suite as its physics) and shadow (the ArWen route's observer on the fixed step).  What stays refused is
    # named below and in lts_v842.admission_refusal.
    from types import SimpleNamespace
    from .config_fullstep_lts import validate_fullstep_lts_request
    from .errors import ConfigurationRefusal
    try:
        validate_fullstep_lts_request(SimpleNamespace(fullstep_local_timestep=True,
            **{name: value for name, value in knobs.items() if value is not None}))
    except ConfigurationRefusal as error:
        return str(error)
    if getattr(arguments, "samehost_devices", None) is not None:
        if knobs["fullstep_lts_mode"] != "active":
            return ("the full-step shadow runs on one device: the partitioned shadow has no owner-only donor "
                    "reduction (halo duplicates would corrupt the rate census)")
        # Retired 2026-10-25 (hex 0.4.6): active LTS on several cards cuts every rate class into blocks, one per
        # card (woof.hex.mpasphys.lts_v842.run_lts_classes on woof.hex.cuda_fullstep_lts_multi).
    return None


def run_forecast(arguments: argparse.Namespace) -> int:
    from woof.output_formats import command_scope
    with command_scope(arguments, engine="woof hex") as scope:
        result = _run_forecast_with_outputs(arguments)
        return scope.finish(result) if scope else result


def _run_forecast_with_outputs(arguments: argparse.Namespace) -> int:
    # The hex default physics (owner ruling 2026-10-25): a run that names no physics takes MPAS-A's own suite on
    # the MPAS route, chosen by mesh (woof.hex.physics_default).  Applied before the roster split, so every member
    # and every segment process decides the same route.  --physics-suite arwen_mpas_column is the ArWen route.
    from . import physics_default
    defaulted = physics_default.apply(arguments)
    if defaulted is not None:
        print(f"woof hex: physics {defaulted['suite']} on the MPAS route (the hex default; "
              f"--physics-suite {physics_default.arwen_suite()} runs WOOF physics)", file=sys.stderr, flush=True)
    if getattr(arguments, "members", None) is not None:
        from .multi_member import run_members
        return run_members(arguments, sys.modules[__name__])
    triggers = getattr(arguments, "nest_triggers", None)
    if triggers is not None:
        # [hex_nests] (woof.hex.nest_triggers): the driver's step loop arms its
        # look from this; refused there by name if the table cannot be read.
        if not Path(triggers).is_file():
            raise ForecastDoorRefusal(f"--nest-triggers {triggers} is not a file")
        os.environ["WOOF_HEX_NEST_TRIGGERS"] = str(Path(triggers).resolve())
    time_ledger.configure(getattr(arguments, "out", None))
    time_ledger.emit("prep.door", "begin")
    time_ledger.emit("forecast.construct", "begin")
    rc = None
    try:
        rc = _run_forecast(arguments)
        return rc
    finally:
        if getattr(arguments, "_native_timeline", False):
            for name in ("forecast.first_step", "forecast.integrate", "forecast.construct", "prep.door"):
                time_ledger.end_if_open(name)
        time_ledger.end_if_open("post.forecast")
        time_ledger.emit("finish", attrs={"scope": "forecast_door", "returncode": rc})


def _mesh_file_spacing_m(arguments: argparse.Namespace) -> float | None:
    """The mesh spacing an MPAS grid or init file records: nominalMinDc, else min(dcEdge); None when neither."""
    import numpy as np
    from woof.netcdf_bridge import open_dataset

    for option in ("init", "grid", "static"):
        value = getattr(arguments, option, None)
        if value is None or not Path(value).is_file():
            continue
        dataset = open_dataset(Path(value))
        for name in ("nominalMinDc", "dcEdge"):
            if name in dataset.variables:
                spacing = float(np.min(np.asarray(dataset.variables[name][:], dtype=np.float64)))
                if spacing > 0.0:
                    return spacing
    return None


def driver_floor_refusal(error: BaseException, admission: Any, driver_argv: Sequence[str]) -> str:
    """The sentence for a run the door admitted and the driver's device-memory floor then refused.

    The door's admission and the driver's floor are ONE number: both are
    :func:`device_admission.required_free_bytes` over the same cells, model and margin (the door passes it
    to the driver as ``--required-free-bytes``).  THE BREAKAGE THIS PREVENTS: this refusal used to say the
    two "disagree" whenever the floor refused, so a card whose free memory fell between the two checks
    (another process allocating) read as a model defect.  Seen 2026-10-16 on a shared 6000 card: the door
    admitted a 44,143-cell cull needing 12,109.1 + 1,756.8 = 13,865.9 MiB with 15,452.4 MiB free; the
    driver then saw 12,048.4 MiB free and refused on 14,539,480,038 B, which is 13,865.9 MiB, the same sum
    printed in bytes.  The sentence now prints both in MiB, says whether they are equal, and names the
    real cause when they are.
    """
    floor = None
    if "--required-free-bytes" in driver_argv:
        try:
            floor = int(driver_argv[list(driver_argv).index("--required-free-bytes") + 1])
        except (IndexError, ValueError):
            floor = None
    required = getattr(admission, "required_bytes", None)
    seen_free = getattr(admission, "free_bytes", None)
    if floor is not None and required is not None and int(floor) == int(required):
        return (f"the card's free device memory fell between the door's admission and the driver's floor: "
                f"both require the same {_mib(floor)} ({floor:,} B, one sum, "
                f"device_admission.required_free_bytes); the door saw {_mib(seen_free or 0)} free and admitted, "
                f"then the driver saw less: {error}  Another process took device memory in between.  Free the "
                f"card (or wait for it) until {_mib(floor)} is free, and re-run.")
    return (f"the driver's own device-memory floor refused this run: {error}  The door admitted on "
            f"{_mib(required or 0)} and the driver's floor asked {_mib(floor or 0)}: the two gates disagree, "
            "which is a defect (they must be one sum, device_admission.required_free_bytes); report both "
            "numbers.")


def prewarm_native_kernels(config=None) -> None:
    """The MPAS suite route's kernel set compiles on every core before step 1 (woof.hex.kernel_plan).

    THE BREAKAGE THIS PREVENTS: the suite route never joined the door's kernel plan, so on a cold kernel
    cache its first physics step compiled every slot's NVRTC image one after another on one core while the
    card waited (x1.2562 convection_permitting, 5090, 2026-10-17: step 1 65 s at ~1 core, steady steps
    under 1 s).  The same recorder and replay the WOOF route uses (kernel_prewarm.install_recorder /
    start_replay) now run here, before the pools are decoded, so the previous run's plan compiles on a pool
    as wide as the usable cores while the host prepares; a run with no plan yet records one.
    """
    from . import kernel_plan, kernel_prewarm

    try:
        kernel_prewarm.install_recorder()
        # Queue order only: a config the order cannot read still gets the whole plan, in its default order
        # (an unreadable config used to skip the replay itself; tests/test_native_kernel_plan_door.py).
        try:
            kernel_plan.prioritize_physics(config)
        except Exception as error:
            print(f"woof hex: kernel plan physics order not applied ({error}); default order kept",
                  file=sys.stderr, flush=True)
        kernel_prewarm.start_replay_async()
    except Exception as error:  # the run compiles as it always did
        print(f"woof hex: kernel plan not started ({error}); kernels compile as first reached",
              file=sys.stderr, flush=True)


def write_kernel_plan_receipt(out: Any) -> None:
    """``kernel-plan-receipt.json`` beside the suite route's receipt: what the plan compiled and what was inline."""
    if out is None:
        return
    try:
        from . import kernel_prewarm

        target = Path(out).expanduser().resolve() / "kernel-plan-receipt.json"
        if target.parent.is_dir():
            target.write_text(json.dumps(kernel_prewarm.receipt(), indent=2, default=str) + "\n")
    except Exception:
        pass


def lts_step(arguments: argparse.Namespace, selection: dict[str, Any], radiation_seconds: float) -> dict[str, Any]:
    """A default step under full-step LTS, moved onto the class clock (:func:`mpas_defaults.lts_class_clock_dt`)."""
    if not getattr(arguments, "fullstep_local_timestep", False):
        return selection
    events = {"radiation": float(radiation_seconds)}
    history = getattr(arguments, "history_every_minutes", None)
    if history:
        events["history"] = float(history) * 60.0
    hours = getattr(arguments, "hours", None)
    if hours:
        events["run length"] = float(hours) * 3600.0
    # The plan's real top class, not the ladder's ceiling (HEX-LAM-VS-ZOOM-2026-10-27 defect 1: the clock took the
    # config dataclass's 16 while both executors plan at most 8, so the 96 -> 3 km zoom globe stepped 12.5 s, not
    # 15 s).  The cap is the executors' own default; the mesh census (woof.hex.fullstep_lts_census) lowers it to the
    # top class the mesh grades to, and the state bounds only lower that further, by powers of two.
    top = lts_top_rate(arguments)
    # lts_choice prices every top class against the fixed step from these (a lower top class may keep the step).
    arguments._lts_clock = {"fixed": dict(selection), "events": dict(events)}
    moved = mpas_defaults.lts_class_clock_dt(selection, max_rate=int(top["rate"]), events=events)
    if moved is not selection:
        moved = {**moved, "lts_top_rate_source": top["source"]}
        print(f"woof hex: timestep {moved['dt_seconds']:g} s for full-step local time stepping (the default "
              f"{selection['dt_seconds']:g} s puts the coarsest class step, {int(top['rate'])} x, off the "
              f"radiation/history clock; {top['source']})", file=sys.stderr, flush=True)
    return moved


def lts_top_rate(arguments: argparse.Namespace) -> dict[str, Any]:
    """The coarsest class full-step LTS can plan on this run's mesh: ``{"rate", "source", "census"}``.

    The cap is ``--fullstep-lts-max-rate``, else the executors' default (8).  When the run's grid (or init) file is
    readable, the mesh's own census lowers it (a mesh whose coarsest cells are 4x its finest never plans 8); the
    census is kept on ``arguments`` for the MPAS route's LTS-or-fixed choice.
    """
    from .fullstep_lts_census import DOOR_DEFAULT_MAX_RATE, mesh_class_census, read_mesh_edges

    cached = getattr(arguments, "_fullstep_lts_top", None)
    if cached is not None:
        return cached
    rate = getattr(arguments, "fullstep_lts_max_rate", None)
    cap = int(DOOR_DEFAULT_MAX_RATE if rate is None else rate)
    source = (f"--fullstep-lts-max-rate {cap}" if rate is not None
              else f"the executors' default top rate {cap}")
    top: dict[str, Any] = {"rate": cap, "source": source, "census": None, "closed": None}
    path = getattr(arguments, "grid", None) or getattr(arguments, "init", None)
    edges = read_mesh_edges(path)
    if edges is not None:
        census = mesh_class_census(*edges, max_rate=cap)
        top = {"rate": max(census), "census": census, "closed": bool((edges[1] >= 0).all()),
               "source": f"{source}; the mesh grades to {max(census)} ({Path(str(path)).name})"}
        arguments._fullstep_lts_edges = edges
    arguments._fullstep_lts_top = top
    return top


def write_lts_choice_receipt(arguments: argparse.Namespace) -> None:
    """``fullstep-lts-choice.json`` beside the run's frames when the door chose between LTS and the fixed step."""
    record = getattr(arguments, "fullstep_lts_door_choice", None)
    out = getattr(arguments, "out", None)
    if record is None or out is None or not Path(out).expanduser().is_dir():
        return
    try:
        (Path(out).expanduser() / "fullstep-lts-choice.json").write_text(json.dumps(record, indent=2) + "\n")
    except OSError:
        pass


def chosen_step(arguments: argparse.Namespace, fixed: dict[str, Any], clocked: dict[str, Any],
                choice: str | None) -> dict[str, Any]:
    """The run's step selection after :func:`lts_choice`: the fixed step's, the clock's, or the chosen class's."""
    if choice == "fixed":
        return dict(fixed)
    step = getattr(arguments, "_lts_chosen_dt", None)
    if choice != "lts" or step is None or float(step) == float(clocked["dt_seconds"]):
        return clocked
    rate = int(getattr(arguments, "fullstep_lts_max_rate", 0) or 0)
    return {**fixed, "dt_seconds": float(step), "lts_max_rate": rate,
            "source": "lts class clock" if float(step) != float(fixed["dt_seconds"]) else fixed.get("source"),
            "default_dt_seconds": float(fixed["dt_seconds"]),
            "rule": (f"{fixed.get('rule', '')}; top class {rate} chosen by the measured cost "
                     f"(woof.hex.fullstep_lts_choice), {rate} x {float(step):g} s on every event")}


def mpas_lts_choice(arguments: argparse.Namespace, *, fixed_dt: float, lts_dt: float, environ=None) -> str | None:
    """:func:`lts_choice` on the MPAS suite route."""
    return lts_choice(arguments, route="mpas-suite", fixed_dt=fixed_dt, lts_dt=lts_dt, environ=environ)


def lts_choice(arguments: argparse.Namespace, *, route: str, fixed_dt: float, lts_dt: float,
               environ=None) -> str | None:
    """Active full-step LTS or the fixed step for this run on ``route``; None when there is nothing to choose.

    THE BREAKAGE THIS PREVENTS (HEX-LAM-VS-ZOOM-2026-10-27 defect 2, HEX-LTS-REGRESSION-2026-10-27): on the
    96 -> 3 km zoom globe, active LTS ran 19 % slower than the fixed step on BOTH physics routes even on the right
    class clock (MPAS 105.9 vs 85.7, ArWen 98.4 vs 79.3 s per forecast hour, RTX 5090): the plan halves the cell
    work per step but steps 1.875 class steps per base step, and a step's fixed host cost is about 90,000 cells'
    worth of work on either route.  The door predicts both from the mesh's class census and the route's measured
    per-step cost (woof.hex.fullstep_lts_choice, data/fullstep-lts-step-cost.v1.json) and runs the fixed step where
    LTS loses, as a plain run: the fixed step's own timestep and no LTS argument left, so its output is the fixed
    run's.  LTS still runs where it wins (a mesh whose cells are mostly coarse).  Chosen only for one card, a closed
    mesh and no moving region (a moving segment needs the class executor's move hooks, and several cards run other
    executors that were not measured).  ``WOOF_HEX_FULLSTEP_LTS_FORCE=1`` runs LTS anyway.
    """
    from .fullstep_lts_admission import moving_launch_switches
    from .fullstep_lts_census import mesh_class_census
    from .fullstep_lts_choice import FORCE_ENV, clock_fits, plan_choice
    from .mpasphys.lts_v842 import lts_requested

    environ = os.environ if environ is None else environ
    if not lts_requested(arguments) or getattr(arguments, "samehost_devices", None) is not None:
        return None
    if moving_launch_switches(environ):
        return None
    top = lts_top_rate(arguments)
    census = top["census"]
    if census is None or len(census) < 2 or not top["closed"]:
        return None  # one class runs the fixed step anyway; a limited-area mesh meets the LTS executor's refusal
    # Every top class from the mesh's own down to 2, each at the step its class clock allows (a default step moves
    # down only as far as that class needs; an explicit step is priced only where it already fits).
    candidates = [(int(top["rate"]), float(lts_dt), census)]
    clock = getattr(arguments, "_lts_clock", None)
    edges = getattr(arguments, "_fullstep_lts_edges", None)
    rate = int(top["rate"]) // 2
    while clock is not None and edges is not None and rate >= 2:
        moved = mpas_defaults.lts_class_clock_dt(clock["fixed"], max_rate=rate, events=clock["events"])
        step = float(moved["dt_seconds"])
        if clock_fits(step, rate, clock["events"]):
            lower = mesh_class_census(*edges, max_rate=rate)
            if len(lower) >= 2:
                candidates.append((rate, step, lower))
        rate //= 2
    choice = plan_choice(sum(int(count) for count in census.values()), route=route, fixed_dt_seconds=fixed_dt,
                         candidates=candidates)
    forced = str(environ.get(FORCE_ENV, "")).strip().lower() in ("1", "true", "yes", "on")
    if forced:
        top_row = choice["candidates"][0]
        choice = {**choice, "forced_by": f"{FORCE_ENV}=1", "choice": "lts", "lts_top_rate": top_row["top_rate"],
                  "lts_dt_seconds": top_row["dt_seconds"]}
    predicted = choice["predicted_seconds_per_forecast_hour"]
    if choice["choice"] == "lts" and int(choice["lts_top_rate"]) != int(top["rate"]):
        # A lower top class is cheaper: the executors plan at most that class, at the step its clock allows.
        arguments.fullstep_lts_max_rate = int(choice["lts_top_rate"])
        print(f"woof hex: local time stepping with top class {int(choice['lts_top_rate'])} at "
              f"{float(choice['lts_dt_seconds']):g} s (predicted {predicted['lts']:g} s per forecast hour; the mesh's "
              f"top class {int(top['rate'])} needs {float(lts_dt):g} s and is predicted slower)", file=sys.stderr,
              flush=True)
    arguments._lts_chosen_dt = None if choice["choice"] != "lts" else float(choice["lts_dt_seconds"])
    lts_dt = float(choice["lts_dt_seconds"] or lts_dt)
    record = {"route": route, "door_choice": choice}
    arguments.fullstep_lts_door_choice = record
    print(json.dumps({"fullstep_lts": record}, default=str), flush=True)
    if choice["choice"] == "fixed":
        for name in ("fullstep_lts_mode", "fullstep_lts_max_rate", "fullstep_lts_buffer_rings"):
            setattr(arguments, name, None)
        arguments.fullstep_local_timestep = False
        print(f"woof hex: fixed step {fixed_dt:g} s instead of active local time stepping: on this mesh LTS is "
              f"predicted at {predicted['lts']:g} s per forecast hour against {predicted['fixed']:g} s for the fixed "
              f"step ({route} route, measured per-step cost); {FORCE_ENV}=1 runs LTS anyway", file=sys.stderr,
              flush=True)
    elif not forced:
        print(f"woof hex: active local time stepping at {lts_dt:g} s: predicted {predicted['lts']:g} s per forecast "
              f"hour against {predicted['fixed']:g} s for the fixed step", file=sys.stderr, flush=True)
    return choice["choice"]


def require_forecast_inputs(arguments: argparse.Namespace, *, native: bool = False,
                            collected: list[str] | None = None) -> tuple:
    """Share file validation and remedies before either route derives a timestep.

    The MPAS route may carry grid/static fields in its init. Restart and DA
    carriers still use that init for static pool fields (checkpoint_v842).
    """
    from .command_names import command_name

    door = command_name()
    requirements = (
        ("grid", "the mesh grid carries the topology; it has no default outside a checkout",
         f"Pass the registered grid file; `{door} mesh-plan` builds a mesh from its spec, "
         f"and `{door} mesh-check` prints its digests."),
        ("static", "the static carries terrain, land use and physical dcEdge",
         "Pass the static generated with this grid. Re-run the preparation command for this mesh "
         f"(`{door} mesh-plan` for a mesh spec, or the original global/regional workflow)."),
        ("init", "the initial condition supplies the forecast state and has no synthetic default",
         f"Build one with `{door} init` (chapter 5 of the manual)."),
    )
    problems = [] if collected is None else collected
    files = []
    for name, breakage, remedy in requirements:
        value = getattr(arguments, name, None)
        if native and name != "init" and value is None:
            files.append(None)
            continue
        value = _require(value, f"--{name}", breakage)
        files.append(_require_file(value, f"--{name}", remedy, problems))
    if collected is None and problems:
        raise _refuse("\n".join(problems))
    return tuple(files)


def bind_native_step(arguments: argparse.Namespace, native_text: str) -> str:
    """The MPAS suite route's ``config_dt``: the namelist's, else the door's own step for the mesh.

    THE BREAKAGE THIS PREVENTS: the suite route took its step only from the namelist, so a run naming a
    suite without a namelist ``config_dt`` (``woof hex forecast --physics-suite convection_permitting``,
    ``woof hex global`` with an MPAS suite) ran at MPAS's Registry default, 720 s, on every mesh: on the
    30 km globe (x1.655362, mesh step 180 s) that is four times the stable step (seen 2026-10-16, 6000
    box: DT_CU = DT_PBL = 720, n_microp = 8).  The step now comes from the rule the WOOF route uses
    (:func:`mpas_defaults.resolve_dt`: ``--dt`` SECONDS, ``row``, or the mesh's own default for its
    spacing, dividing the radiation interval).  A namelist ``config_dt`` still decides, and an explicit
    ``--dt`` naming a different step is refused as two steps for one run.
    """
    from .mpasphys.forecast_v842 import namelist_option, set_namelist_options
    from .mpasphys.driver_v842 import interval_seconds

    requested = getattr(arguments, "dt", None)
    named = namelist_option(native_text, "nhyd_model", "config_dt")
    if named is not None:
        text = None if requested is None else str(requested).strip().lower()
        if text not in (None, "", "auto", "row") and float(text) != float(named):
            raise ValueError(f"--dt {requested} and the namelist config_dt {named} name two steps for one run")
        mpas_lts_choice(arguments, fixed_dt=float(named), lts_dt=float(named))
        return native_text
    registry = load_registry(resolve_repo(getattr(arguments, "repo", None)))
    row = registry.get(getattr(arguments, "mesh", None))
    nominal_dx_m = None if getattr(row, "nominal_dx_m", None) is None else float(row.nominal_dx_m)
    row_dt_seconds = None if getattr(row, "dt_seconds", None) is None else float(row.dt_seconds)
    if row is None:
        # An unregistered mesh (a plain MPAS grid named by --mesh) takes the spacing its own files record:
        # MPAS's nominalMinDc, else the smallest dcEdge.  THE BREAKAGE THIS PREVENTS: a bare suite run on
        # x1.2562 refused "not a registered mesh" although its init names its spacing (2026-10-17).
        nominal_dx_m = _mesh_file_spacing_m(arguments)
        if nominal_dx_m is None:
            raise ValueError(f"--mesh {getattr(arguments, 'mesh', None)!r} is not a registered mesh and its files "
                             "carry no nominalMinDc or dcEdge, so the MPAS suite route has no step for it; name "
                             "--dt SECONDS or a namelist config_dt")
    radiation = namelist_option(native_text, "physics", "config_radtlw_interval")
    radiation_seconds = (float(interval_seconds(radiation)) if radiation not in (None, "", "none")
                         else float(mpas_defaults.DEFAULT_RADIATION_SECONDS))
    fixed = mpas_defaults.resolve_dt(
        requested, nominal_dx_m=nominal_dx_m, row_dt_seconds=row_dt_seconds,
        radiation_seconds=radiation_seconds)
    selection = lts_step(arguments, fixed, radiation_seconds)
    # "fixed": the plain fixed-step run (the step a run without LTS takes, no LTS argument left for any reader);
    # "lts": the step of the top class the choice took.
    selection = chosen_step(arguments, fixed, selection,
                            mpas_lts_choice(arguments, fixed_dt=float(fixed["dt_seconds"]),
                                            lts_dt=float(selection["dt_seconds"])))
    dt = float(selection["dt_seconds"])
    print(f"woof hex: MPAS suite timestep {dt:g} s ({selection.get('source')}: {selection.get('rule', '')})",
          file=sys.stderr, flush=True)
    return set_namelist_options(native_text, {"nhyd_model": {"config_dt": dt}})


#: The door's dycore schedule flags and the MPAS &nhyd_model option each one names.
NATIVE_SCHEDULE_FLAGS = (
    ("acoustic_substeps", "--acoustic-substeps", "config_number_of_sub_steps"),
    ("time_integration_order", "--time-integration-order", "config_time_integration_order"),
    ("monotonic", "--monotonic", "config_monotonic"),
)


def _namelist_value(value: Any) -> Any:
    """A parsed namelist scalar as an int or bool (Fortran logicals arrive as text from some readers)."""
    if isinstance(value, str):
        text = value.strip().strip(".").lower()
        if text in ("true", "t"):
            return True
        if text in ("false", "f"):
            return False
        return int(text)
    return value


def bind_native_dynamics(arguments: argparse.Namespace, native_text: str) -> str:
    """The MPAS suite route's dycore schedule: each door flag given becomes its &nhyd_model option.

    THE BREAKAGE THIS PREVENTS (2026-10-27, time-step options on the 96->3 km zoom globe): this route returns
    before resolve_request, the only reader of --acoustic-substeps, --time-integration-order and --monotonic,
    so `--dt 24 --acoustic-substeps 6` ran four acoustic sub-steps (the history's config_number_of_sub_steps
    was 4) and nothing said the flag had been dropped.  A flag not given leaves the namelist (or MPAS's
    Registry default) in charge, as before; a namelist naming a different value is refused as two values
    for one run, as bind_native_step refuses two steps.
    """
    from .mpasphys.forecast_v842 import namelist_option, set_namelist_options

    options: dict[str, Any] = {}
    for attribute, flag, option in NATIVE_SCHEDULE_FLAGS:
        given = getattr(arguments, attribute, None)
        if given is None:
            continue
        value = bool(given) if attribute == "monotonic" else int(given)
        named = namelist_option(native_text, "nhyd_model", option)
        if named is not None:
            if _namelist_value(named) != value:
                raise ValueError(f"{flag} {given} and the namelist {option} {named} name two values for one run")
            continue
        options[option] = value
    if not options:
        return native_text
    print("woof hex: MPAS suite dycore schedule " + ", ".join(f"{k}={v}" for k, v in options.items())
          + " (explicit flags)", file=sys.stderr, flush=True)
    return set_namelist_options(native_text, {"nhyd_model": options})


def _run_forecast(arguments: argparse.Namespace) -> int:
    """The ``woof hex forecast`` handler.  Returns the process exit code."""

    from .mpasphys.forecast_v842 import native_route, run_forecast as run_native_physics
    # --analysis (lane W1): check the bundle against the run and rewrite it into the door's own
    # inputs (--init, or the background with an IAU window, or a DA-cycling restart), so the
    # forecast below is the code a plain --init run executes.
    from .analysis_door import AnalysisDoorRefusal, resolve_analysis

    try:
        # The registry is read after the door binds a bundle's carried mesh, whose row it adds.
        arguments.analysis_receipt = resolve_analysis(
            arguments, registry_loader=lambda: load_registry(resolve_repo(getattr(arguments, "repo", None))))
    except AnalysisDoorRefusal as error:
        raise _refuse(str(error)) from None
    try:
        # An MPAS suite route that names no mode runs correct (owner ruling 2026-10-13: correct mode is the
        # standard; coordinator 2026-10-14: default correct, gaps listed loudly, never refused).
        # --physics-mode exact runs native MPAS v8.4.2 bit for bit; every proof names it.
        native_config, native_text = native_route(arguments, default_mode=PHYSICS_MODE_DEFAULT)
        require_forecast_inputs(arguments, native=native_config is not None)
        if native_config is not None:
            from .mpasphys.resident_v842 import ported_route_refusal
            from .mpasphys.forecast_v842 import named_library_route
            suite_refusal = ported_route_refusal(native_config)
            # A run that names a native MPAS library runs this selection on it (forecast_v842.named_library_route).
            if suite_refusal is not None and not named_library_route(arguments, native_config):
                raise ValueError(suite_refusal)
            # A moving region on this route runs through its own move hooks (woof.hex.mpasphys.moving_v842).
            lts_refusal = mpas_suite_lts_refusal(arguments)
            if lts_refusal is not None:
                raise ValueError(lts_refusal)
            # Retired (hex parity run group 7, #25): --balance-series is sampled on this route too
            # (mpasphys.runtime_v842, the same woof.hex.iau_forcing.BalanceSeries on the dycore's own state).
            if getattr(arguments, "source_table", None) is not None:
                # THE BREAKAGE THIS PREVENTS (hex parity run group 7, #27): this route returns before
                # resolve_request, the only reader of --source-table, so `pair --physics-suite ...` ran a treatment
                # leg byte-identical to its control (80 of 80 history variables) and no receipt named the table.
                # The ArWen route refuses the table the same way on every in-tree row (resolve_run_row).
                raise ValueError(
                    f"--source-table {arguments.source_table} was given, but the MPAS suite route carries no "
                    "point source and no seam to release one into; the table would be accepted and never read. "
                    "A point source runs on a --physics-backend row that declares one (none of the in-tree rows "
                    "does), without --physics-suite")
            # Retired (named gap W1-G2): the suite route applies IAU now (mpasphys.runtime_v842.SuiteIau, MPAS's
            # atm_add_tend_anal_incr once per step).  The door's IAU flags become MPAS's own &iau group and iau
            # stream input, so a flag run and a namelist run execute the same code.
            if getattr(arguments, "iau_window_seconds", None) is not None:
                from .mpasphys.forecast_v842 import bind_door_iau
                native_text = bind_door_iau(arguments, native_text)
            # Upstream code under restricted terms (Noah-MP v5.0: UCAR forbids sale) runs only by an opt-in that names
            # its terms (woof.hex.licence_gate; owner ruling 2026-10-14).  No default selects it.  UGWP and the RRFS
            # fork's NOAA GSL physics are not gated (owner ruling 2026-10-14: treated as US Government code).
            from . import licence_gate
            native_config = licence_gate.admit(native_config, arguments)
            announce_physics_mode(native_config, arguments)
            announce_cpu_slots(native_config)
            prewarm_native_kernels(native_config)
            native_text = bind_native_step(arguments, native_text)
            native_text = bind_native_dynamics(arguments, native_text)
            if getattr(arguments, "orographic_statistics", None) is None:
                arguments.orographic_statistics = mpas_route_orographic_statistics(arguments, native_config)
            arguments._native_timeline = True
            time_ledger.emit("prep.door", "end")
            try:
                rc = run_native_physics(arguments, native_config, native_text)
                if rc == 0:
                    native_default_products(arguments)
                    receipt = getattr(arguments, "receipt", None)
                    if receipt is not None:
                        # THE BREAKAGE THIS PREVENTS (2026-10-25, MPAS physics the hex default): this route returns
                        # before resolve_request, the only reader of --receipt, so the flag was accepted and no file
                        # was written (a cycle slot's forecast-receipt.json never appeared).  The route's receipt is
                        # native-physics-receipt.json beside its frames; --receipt gets that receipt.
                        import shutil
                        source = Path(arguments.out) / "native-physics-receipt.json"
                        if not source.is_file():
                            raise RuntimeError(f"--receipt {receipt}: the MPAS suite route finished without "
                                               f"{source.name} in {arguments.out}")
                        shutil.copyfile(source, Path(receipt).expanduser())
                return rc
            finally:
                write_kernel_plan_receipt(getattr(arguments, "out", None))
                write_lts_choice_receipt(arguments)
    except (ValueError, OSError, RuntimeError) as error:
        raise _refuse(str(error)) from error
    started = time.monotonic()
    request = resolve_request(arguments)
    from .fullstep_lts_admission import forecast_moving_route, moving_route_refusal

    moving_route = forecast_moving_route(
        fullstep_local_timestep=bool(request.fullstep_local_timestep), fullstep_lts_mode=request.fullstep_lts_mode,
        fullstep_lts_max_rate=request.fullstep_lts_max_rate,
        multi_card=getattr(arguments, "samehost_devices", None) is not None)
    moving_refusal = None if moving_route is None else moving_route_refusal(moving_route)
    if moving_refusal is not None:
        raise _refuse(moving_refusal)
    from .speculative_contract import validate_output
    validate_output(request.out)
    registry = load_registry(request.repo)
    samehost_options = None
    samehost = None
    if getattr(arguments, "samehost_devices", None) is not None:
        samehost = importlib.import_module("woof.hex.drivers.samehost_forecast_v841")
        try:
            samehost_options = samehost.request_options(arguments, request)
        except (ValueError, OSError, RuntimeError) as error:
            raise _refuse(str(error)) from error
    else:
        samehost = importlib.import_module("woof.hex.drivers.samehost_forecast_v841")
        try:
            samehost_options = samehost.automatic_options(arguments, request)
        except (ValueError, OSError, RuntimeError) as error:
            raise _refuse(str(error)) from error
    if samehost_options is not None and len(set(samehost_options["devices"])) > 1:
        # Ranks on distinct cards share this process: the engine's land kernels are held per card before
        # any is compiled (woof.hex.engine_per_card names the breakage).
        from .engine_per_card import bind_engine_modules_per_card
        try:
            bind_engine_modules_per_card(importlib.import_module("cupy"))
        except RuntimeError as error:
            raise _refuse(str(error)) from error
    if (request.fullstep_local_timestep and samehost_options is not None and request.fullstep_lts_max_rate != 1
            and request.fullstep_lts_mode != "active"):
        raise _refuse("the full-step shadow runs on one device: the partitioned shadow has no owner-only donor reduction (halo duplicates would corrupt the rate census)")
    if request.preflight:
        if samehost_options is not None:
            admission = samehost.admit_request(request, registry, samehost_options, sys.modules[__name__])
            require_admitted(admission)
            binding, driver = _load_drivers(request.repo)
            _bind(binding, driver, request)
            rc = int(driver.main(build_driver_argv(request)))
            print(json.dumps({"status": "samehost_preflight_passed" if rc == 0 else "samehost_preflight_failed",
                              "admission": samehost_options["admission_records"],
                              "layouts_verified": True, "device_stacks_constructed": False,
                              "identity_checked": False}, default=str), flush=True)
            return rc
        return _run_preflight(request, registry, started)

    # The point chain selects its suite after the legacy init step. Resolve
    # soil from the recorded MET and complete aerosols on the ordinary road.
    from .init_physics import prepare_forecast_init
    from dataclasses import replace
    import subprocess
    try:
        prepared, init_receipt = prepare_forecast_init(
            request.init, request.out.with_name(request.out.name + ".physics-init.nc"),
            request.physics_resolution, request.start_time)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise _refuse(str(error)) from error
    if init_receipt is not None:
        request = replace(request, init=prepared)
        print(json.dumps({"physics_initialization": init_receipt}), flush=True)

    # Admission first on a real run: it is the cheapest of the remaining
    # checks and the most consequential, and refusing here spends neither
    # the mesh hashing nor a CUDA context on a run that cannot start.
    if samehost_options is None:
        request = _with_resolved_model(request)
        admission = admit_device(
            mesh=request.mesh, cells=request.cells, headroom_bytes=request.headroom_bytes,
            registry=registry, model=request.model,
            stochastic=request.stochastic,
        )
    else:
        admission = samehost.admit_request(request, registry, samehost_options, sys.modules[__name__])
        request = _pinned_for_driver(request)
    _print_admission(request, admission)
    if (not admission.admitted and samehost_options is None
            and os.environ.get("WOOF_HEX_MOVING_CHECKPOINT_WAIT")):
        # A pre-launched move segment starts while the member it replaces still
        # holds the card, so the free memory measured now undercounts what this
        # run will have.  The decision is deferred, not dropped: it is taken
        # again, against the card as it then is, once the carrier is sealed and
        # before any device state is built (woof.hex.moving_control.wait_for_carrier),
        # and it refuses by the same sentence if the card does not free in time.
        from . import moving_control

        def _readmit(request=request):
            verdict = admit_device(
                mesh=request.mesh, cells=request.cells, headroom_bytes=request.headroom_bytes,
                registry=registry, model=request.model,
                stochastic=request.stochastic,
            )
            return verdict

        def _require(verdict, request=request):
            _print_admission(request, verdict)
            require_admitted(verdict)

        moving_control.defer_admission(_readmit, _require)
        print("ADMISSION DEFERRED pre-launched move segment: decided again once the "
              "carrier is sealed and the previous member has released the card", flush=True)
    elif not admission.admitted:
        receipt = build_receipt(
            request=request, admission=admission, bind_receipt=None,
            driver_receipt=None, history=[], driver_argv=[],
            seconds=time.monotonic() - started, status="refused_by_admission",
        )
        written = _write_receipt(request, receipt)
        if written is not None:
            print(f"RECEIPT {written}", flush=True)
        require_admitted(admission)

    # The architecture half of admission, answered before a CUDA context or
    # a kernel compile is spent on a card the run would refuse by name.
    # Same-host admission records the architecture of each assigned card
    # in the driver receipt rather than one process-wide device record.
    architecture: dict[str, Any] | None = None
    if samehost_options is None:
        architecture = admit_architecture()
        print(
            f"ARCHITECTURE sm={architecture['sm']} -> admitted "
            f"({architecture['basis']})", flush=True,
        )
        # Compile the previous run's CUDA modules on a thread pool while the
        # driver reads and prepares host state (woof.hex.kernel_prewarm); the
        # forecast then finds them in CuPy's own disk cache.
        from . import kernel_prewarm

        kernel_prewarm.install_recorder()
        kernel_prewarm.start_replay_async()
    else:
        # Same-host ranks find their images in the same CuPy disk cache the
        # replay pool writes (CuPy 14 binds its cache directory once, when it
        # is imported, so every rank and the pool share it), so the one-card
        # replay serves them too: open the first card as admit_architecture
        # does for one card, then compile the previous run's modules on the
        # pool while the host prepares.  Breakage it prevents: a multi-card
        # run never recorded or replayed, so on a box with an empty kernel
        # cache every compile ran on the rank threads inside step 1 (361.5 s
        # on the first 20 km run on a rented box, 2026-10-13).
        from . import kernel_prewarm
        from .cuda_backend.runtime import require_cuda

        try:
            first = int(samehost_options["devices"][0])
            require_cuda(device_id=first, min_compute=(12, 0))
        except Exception:
            pass  # no replay; each rank refuses by name, or compiles, as before
        else:
            kernel_prewarm.install_recorder()
            kernel_prewarm.start_replay_async(device_id=first)

    binding, driver = _load_drivers(request.repo)
    bind_receipt = _bind(binding, driver, request)
    driver_argv = build_driver_argv(request)
    time_ledger.emit("prep.door", "end")
    # The driver's validate_destination requires BOTH roots absent and creates
    # them itself; a door that pre-creates them kills every admitted run with
    # FileExistsError immediately after admission (measured on the RTX 3080,
    # 2026-08-24 -- the door's success leg had never run on hardware).  The
    # door's own earlier refusals still catch a pre-existing --out/--scratch
    # by name; creation belongs to the driver alone.
    try:
        if samehost_options is None:
            rc = int(driver.main(driver_argv))
        else:
            # A rank process rebuilds this process's module state from the same request and driver argv
            # (samehost_rank_group_v841.bootstrap): the mesh bind and the physics resolution are replayed there.
            with samehost.installed(driver, **samehost_options,
                                    door_replay={"repo": request.repo, "request": request,
                                                 "driver_argv": list(driver_argv)}):
                rc = int(driver.main(driver_argv))
    except MemoryError as error:
        raise _refuse(driver_floor_refusal(error, admission, driver_argv)) from None
    except ForecastDoorRefusal:
        raise
    except Exception as error:
        # The CAUSE survives: a transactional driver raises its own summary
        # ("composite step aborted without publication") over a chained
        # exception, and a refusal that carries only the summary is a wall
        # with no sign.  The whole chain is written beside the run.
        import traceback

        trace_path = request.out / "forecast-traceback.log"
        cause_note = ""
        try:
            request.out.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            cause = error.__cause__ or error.__context__
            if cause is not None:
                cause_note = f"  Underlying cause: {type(cause).__name__}: {cause}"
            cause_note += f"  Full traceback: {trace_path}."
        except OSError:
            pass
        raise _refuse(
            f"the forecast driver stopped with {type(error).__name__}: {error}"
            f"{cause_note}"
            f"  The scratch tree is kept at {request.scratch} and any frames "
            f"already written are in {request.out}."
        ) from None

    history = history_files(request.out)
    driver_receipt: dict[str, Any] | None = None
    driver_receipt_path = request.out / DRIVER_RECEIPT_NAME
    if driver_receipt_path.is_file():
        try:
            driver_receipt = json.loads(
                driver_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            driver_receipt = {"unreadable": str(driver_receipt_path)}

    status = "passed" if rc == 0 else "driver_rc_nonzero"
    if driver_receipt is not None and driver_receipt.get("status"):
        status = str(driver_receipt["status"])
    receipt = build_receipt(
        request=request,
        admission=admission,
        bind_receipt=bind_receipt,
        driver_receipt=driver_receipt,
        history=history,
        driver_argv=driver_argv,
        seconds=time.monotonic() - started,
        status=status,
        architecture=architecture,
    )
    observation_path = request.out / "observation-scorecard" / "scorecard.json"
    if observation_path.is_file():
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        receipt["observation_scorecard"] = {
            "status": observation["status"],
            "path": str(observation_path),
            "measured_metrics": observation.get("measured_metrics", 0),
        }
    written = _write_receipt(request, receipt)
    if written is not None:
        print(f"RECEIPT {written}", flush=True)
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))

    print(
        f"DOOR mesh={request.mesh} steps={request.steps} "
        f"frames={len(history)} status={status} out={request.out}",
        flush=True,
    )
    write_lts_choice_receipt(arguments)
    command = render_command(request, history)
    if command is not None and rc == 0 and not getattr(arguments, "no_products", False):
        # The standard product set is the forecast's last stage by default
        # (fixed means default; ship only what users can reach): a bare run
        # ends with maps, soundings and meteograms on disk, not a command to
        # copy.  A product failure is reported by name and leaves the
        # forecast's own exit code alone -- the frames are good either way.
        run_default_products(request, history, arguments)
    elif command is not None:
        print("NEXT " + " ".join(command), flush=True)
    elif rc == 0:
        print(
            "DOOR the driver exited 0 but wrote no history frame; the "
            "receipt's driver_receipt block carries the reason",
            flush=True,
        )
        return 1
    return rc


# ---------------------------------------------------------------------------
# the argument surface
# ---------------------------------------------------------------------------
def add_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    from woof.output_formats import add_arguments
    add_arguments(parser)
    parser.add_argument("--fullstep-local-timestep", action=argparse.BooleanOptionalAction, default=False,
                        help="Full-step local time stepping. Mode shadow (the default mode) only measures: "
                             "fixed numerics remain in use. Mode active steps every rate class of a "
                             "variable mesh at its own whole step, physics included. Uses one card "
                             "unless --samehost-devices explicitly requests multi-card active LTS")
    parser.add_argument("--fullstep-lts-mode", choices=("shadow", "active"), default=None)
    parser.add_argument("--fullstep-lts-max-rate", type=int, choices=(1,2,4,8,16,32), default=None)
    parser.add_argument("--fullstep-lts-buffer-rings", type=int, default=None)
    parser.add_argument("--boundary-fingerprint-every", type=int, default=0, metavar="STEPS",
                        help="record the existing committed-state fingerprints every N steps (0 disables)")
    """Wire the ``forecast`` subcommand's arguments onto ``parser``.

    Nothing is ``required=True``.  argparse's own missing-argument error is a
    usage block, and this project's doors state the WRONG RESULT a missing
    argument would cause; the checks live in :func:`resolve_request` so every
    one of them can be exercised without a card.
    """

    parser.add_argument("--samehost-devices", default=None, metavar="D0,D1,...",
                        help="explicitly select resident cards, including opt-in multi-card active LTS; "
                             "repeated ordinals check same-card rank identity")
    parser.add_argument("--samehost-layouts", type=Path, default=None,
                        help="N-rank K=3 partition layouts with native regional padding")
    parser.add_argument("--samehost-part", type=Path, default=None,
                        help="partition file whose hash the layouts carry")
    parser.add_argument("--samehost-transport", choices=("auto", "peer", "staged"), default="auto",
                        help="same-host halo copy route")
    parser.add_argument("--samehost-ranks", choices=("auto", "processes", "threads"), default=None,
                        help="run each card's rank as its own process (auto: the default on Linux, on both physics "
                             "routes, so the ranks' host work runs on one core each on any Python) or as threads of "
                             "this process (threads; with the GIL their host work shares one core)")
    parser.add_argument("--samehost-reference", type=Path, default=None,
                        help="single-card boundary fingerprints for the identity gate")
    parser.add_argument("--samehost-fingerprint-every", type=int, default=None,
                        help="identity boundary interval in steps; zero records only initial and final boundaries. "
                             "Default: every step with --samehost-reference (the identity gate needs every "
                             "boundary), otherwise zero")
    parser.add_argument(
        "--mesh", default=None, metavar="NAME",
        help="registered mesh row (src/hexcore/drivers/mpas_mesh_binding.py)")
    parser.add_argument(
        "--grid", type=Path, default=None, metavar="FILE",
        help="the row's grid netCDF; pinned by byte count and SHA-256")
    parser.add_argument(
        "--static", type=Path, default=None, metavar="FILE",
        help="the row's static netCDF, generated with that grid")
    parser.add_argument(
        "--init", type=Path, default=None, metavar="FILE",
        help="initial conditions from `woof hex init`")
    parser.add_argument(
        "--init-source", default=None, metavar="TEXT",
        help="provenance sentence for the init, recorded in the receipt "
             '(e.g. "GFS 2026-08-24 00Z")')
    parser.add_argument(
        "--invariant", type=Path, default=None, metavar="FILE",
        help="the native invariant stream the vertical grid is read from, as "
             "MPAS-A does when a run configures one (default: the file the "
             "mesh row declares, when it sits beside --grid or --static; "
             "RRFS v2's conus12km and conus3km declare one)")
    parser.add_argument(
        "--no-invariant", action="store_true",
        help="read the vertical grid from the init file even when the mesh "
             "row's invariant stream is present (reproduces runs made before "
             "the invariant default)")
    parser.add_argument(
        "--lbc-dir", type=Path, default=None, metavar="DIR",
        help="lateral-boundary files (lbc.*.nc from `rw_mpas_lbc`) for a "
             "LIMITED-AREA grid. THE BREAKAGE IT PREVENTS: a grid cut with "
             "`rw_mpas_mesh --cull-parent` carries a seven-ring boundary "
             "zone whose outermost cells hold no atmosphere of their own; "
             "integrated with nothing driving them the domain empties from "
             "the edge inward. Required when --grid carries the bdyMask "
             "triple, refused when it does not")
    parser.add_argument(
        "--physics-backend", default=DEFAULT_BACKEND, metavar="ROW",
        help="mpasphys selects the MPAS suite route (its Registry default is "
             "mesoscale_reference; --physics-suite or a namelist may select another). "
             "Otherwise, the WOOF-route column-physics backend row this run selects "
             "(woof.hex.physics_backend_admission); naming a row selects the WOOF "
             "route. Its default is the frozen lane. A provider's row runs its own pinned column "
             "batch and may declare extra prognostic scalars and a point "
             "source; an unregistered name is refused naming the rows "
             "this installation carries")
    parser.add_argument("--physics-profile", help="opt-in MPAS experiment row (for example hrrr-like); defaults stay unchanged")
    parser.add_argument(
        "--physics-suite", default=None, metavar="NAME",
        help="named physics suite (`woof hex physics list`). With no physics "
             "named (no suite, namelist, --physics override, --convection scheme or "
             "backend row) the run takes the hex default: MPAS-A's own physics on "
             "the MPAS route, the suite picked for the mesh (convection_permitting; "
             "data/profiles/physics-default.v1.json). arwen_mpas_column runs WOOF "
             "physics exactly as a bare run did before 0.4.6; arwen_thompson_column "
             "and rrfs_v2_dev are the other WOOF-route suites. MPAS's own suite names "
             "(mesoscale_reference, convection_permitting, hrrrv5) run the MPAS route, "
             "and an option hex cannot run is refused by name")
    parser.add_argument(
        "--physics", action="append", default=None, metavar="KEY=VALUE",
        help="override one physics option, repeatable: an MPAS selector "
             "(config_microp_scheme, config_convection_scheme, config_pbl_scheme, "
             "config_sfclayer_scheme, config_lsm_scheme, config_gwdo_scheme, "
             "config_radt_lw_scheme, config_radt_sw_scheme, config_radt_cld_scheme), "
             "a WOOF generation selector (thompson_version, thompson_fork_snow_fall, "
             "bl_mynn_version, bl_mynn_mixlength, mynn_sfclay_variant, "
             "rrtmg_cloud_optics_form) or config_frac_seaice. Recorded as an "
             "override in the receipt. With no --physics-suite it overrides the WOOF "
             "suite (arwen_mpas_column); name an MPAS suite to override on the MPAS route")
    parser.add_argument(
        "--physics-namelist", type=Path, default=None, metavar="FILE",
        help="an MPAS namelist.atmosphere whose physics the run takes "
             "(config_physics_suite, the scheme selectors, num_soil_layers, "
             "config_frac_seaice, the radiation interval). Every option is "
             "admitted, ignored as MPAS ignores it, or refused by name; a "
             "refused option refuses the run unless --physics overrides it. "
             "`woof hex physics import FILE` prints the verdict per option")
    parser.add_argument(
        "--physics-mode", choices=("correct", "exact"), default=None,
        help="physics.mode on every route. 'correct' (the default when the run names no mode, owner "
             "ruling 2026-10-13: correct mode is the standard) applies the fixes this tree and the engine "
             "carry: on an MPAS suite route (--physics-suite or the namelist's config_physics_suite) the "
             "fixes for native MPAS v8.4.2 defects (woof.hex.mpasphys.driver_v842.CORRECTIONS_V842); on the "
             "WOOF route the engine's corrections (woof.hex.correct_mode: WB4 New Tiedtke dry-mass "
             "coupling, WB5 WSM6 conservative surface integral).  A finding the run cannot apply is listed "
             "as a named gap in the receipt (correct_mode.gaps) and printed at startup, never refused. "
             "'exact' reproduces the matched arithmetic bit for bit; every identity proof names it. "
             "Same key as the namelist's config_physics_mode in &nhyd_model")
    parser.add_argument(
        "--physics-pin-set", default=None,
        help="source pin set for an MPAS physics suite; omitted uses the suite route default")
    parser.add_argument(
        "--physics-default-backend", choices=("hex-cuda", "native-cpu"), default=None,
        help="the backend every MPAS-suite physics slot takes unless its config_*_backend names one. "
             "Without it each route takes its own: hex-cuda on the MPAS-A v8.4.2 suites, so every slot runs "
             "its GPU port, and a slot this install cannot run on the GPU runs MPAS's own Fortran on the CPU "
             "and is named at start-up (PHYSICS ON CPU) and in the receipt (physics.cpu_slots). native-cpu "
             "runs MPAS's own Fortran in every slot: the native referee")
    parser.add_argument(
        "--physics-correct-skip", action="append", default=None, metavar="FINDING",
        help="leave one correction out of a correct-mode run (an observation A/B arm), by finding id "
             "(woof.hex.correct_mode.known_findings: the MPAS driver's CORRECTIONS_V842 and the WOOF "
             "route's engine findings); repeatable or comma-separated. Same key as the namelist's "
             "config_physics_correct_skip")
    parser.add_argument(
        "--licence-opt-in", action="append", default=None, metavar="TERMS",
        help="run an MPAS suite selection whose upstream code is under terms that do not permit an Apache-2.0 "
             "distribution that may be sold, naming those terms (woof.hex.licence_gate; "
             "data/profiles/licence-gates.v1.json): 'ucar-noahmp' for Noah-MP v5.0 "
             "(config_lsm_scheme='sf_noahmp' on an MPAS suite or the RRFS stage; UCAR's terms forbid selling it "
             "or any work containing it). Repeatable or comma-separated; WOOF_HEX_LICENCE_OPT_IN does the same. "
             "Without it such a run is refused by name; no default selects one. Recorded in the receipt "
             "(physics.licence_opt_in). UGWP and the RRFS stage's NOAA GSL physics need no opt-in; the retired "
             "names 'ugwp-held' and 'noaa-gsl-unlicensed' are accepted and do nothing")
    parser.add_argument("--stream-list-directory", type=Path, default=None,
                        help="directory of the native stream field-list files")
    parser.add_argument("--streams", type=Path, default=None,
                        help="MPAS streams.atmosphere for the explicit MPAS physics route")
    parser.add_argument("--stream-input", action="append", default=None, metavar="NAME=FILE",
                        help="bind an MPAS input stream to a file (repeatable)")
    parser.add_argument("--native-physics-library", type=Path, default=None,
                        help="strict test-only MPAS physics library and ABI receipt")
    parser.add_argument("--native-physics-tables", type=Path, action="append", default=None,
                        help="additional pinned native table directory (repeatable)")
    parser.add_argument("--physics-resources", type=Path, default=None,
                        help="certified immutable resources for selected MPAS CUDA slots")
    parser.add_argument("--native-pool-reader", type=Path, default=None,
                        help="Rust MPAS Registry pool transport executable")
    parser.add_argument("--mpas-source", type=Path, default=None,
                        help="pinned read-only MPAS v8.4.2 source (Registry authority)")
    parser.add_argument(
        "--orographic-statistics", type=Path, default=None, metavar="FILE",
        help="the mesh's GSL drag statistics (<mesh>.ugwp_oro_data.nc), read "
             "when the physics selects config_gwdo_scheme='bl_ugwp_gwdo' "
             "(default: the file of that name beside --grid)")
    parser.add_argument("--chemistry-config", type=Path, default=None,
                        help="N aerosol species and GPU mixing/deposition/settling inputs (chemistry/v1 JSON)")
    parser.add_argument("--wind-farms", default=None, metavar="uswtdb|TABLE|off",
                        help="opt-in Fitch turbine drag and TKE from the shared turbine tables")
    parser.add_argument(
        "--source-table", type=Path, default=None, metavar="FILE",
        help="a point-source table the selected row's seam releases from "
             "(one waypoint per line); recorded in the receipt and pinned "
             "by its bytes into the run's identity. Refused on a row that "
             "carries no point source, because the table would be "
             "accepted and never read")
    parser.add_argument(
        "--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="asserted against the init's config_start_time; the init is "
             "the authority")
    parser.add_argument(
        "--hours", type=float, default=None, metavar="H",
        help="forecast length; must be a whole number of the row's timesteps")
    parser.add_argument(
        "--dt", default=None, metavar="{auto,row,SECONDS}",
        help="model timestep. 'auto' (the default) is MPAS-A's rule, 6 s per "
             "km of the mesh's nominalMinDc rounded down to divide the "
             "radiation interval (15 km -> 90 s, 30 km -> 180 s, 60 km -> "
             "360 s); 'row' is the timestep the mesh row was registered "
             "with; a number is used as given. The bind measures it against "
             "the mesh's real dcEdge and refuses only an unstable step")
    parser.add_argument(
        "--time-integration-order", type=int, choices=(2, 3), default=None,
        help="RK order (default 2, MPAS-A Registry.xml:69)")
    parser.add_argument(
        "--acoustic-substeps", type=int, default=None,
        help="acoustic sub-steps per RK step (default 4, MPAS-A v8.4.2 "
             "Registry.xml:104; 2 was the default through v8.4.1 and costs "
             "less, at the price of small-scale noise in the jets)")
    parser.add_argument(
        "--radiation-seconds", type=float, default=None,
        help="radiation cadence in seconds (default 1800, MPAS-A's "
             "config_radtlw/sw_interval 00:30:00, or the "
             "--physics-namelist's radiation interval)")
    parser.add_argument(
        "--monotonic", action=argparse.BooleanOptionalAction, default=None,
        help="the native flux-corrected scalar limiter (default on, MPAS-A "
             "Registry.xml:239); --no-monotonic turns it off")
    parser.add_argument("--cam-coefficient",type=float,default=0.0)
    parser.add_argument("--cam-damping-levels",type=int,default=4)
    parser.add_argument("--rayleigh-u",action="store_true")
    parser.add_argument("--damping-start-m",type=float,default=22000.0)
    parser.add_argument(
        "--max-wind", type=float, default=None, metavar="M_PER_S",
        help="the largest horizontal wind the timestep stability estimate "
             "assumes (default 125 m/s, or the mesh row's own bound).  The "
             "estimate does not read the case's winds; state a lower bound "
             "to run a longer step, a higher one for a stricter check")
    parser.add_argument(
        "--dynamics", action="append", default=None, metavar="KEY=VALUE",
        help="set a dynamics option that has no flag of its own, by its "
             "MPAS namelist name (repeatable): config_apvm_upwinding=0.25, "
             "config_smdiv=0.2, config_coef_3rd_order=1.0, config_xnutr=0.2, "
             "config_positive_definite=true, config_smagorinsky_coef=0.125.  "
             "A --physics-namelist's dynamics options are taken the same "
             "way; a value given both ways must agree")
    parser.add_argument(
        "--history-every-minutes", type=float, default=None, metavar="M",
        help="history cadence; must divide the run into whole steps")
    parser.add_argument("--history-from-minutes", type=float, default=0.0,
                        help="write scheduled history frames starting at this lead minute")
    parser.add_argument(
        "--fingerprint-every", type=int, default=0, metavar="STEPS",
        help="boundary identity fingerprint interval; zero disables recording")
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="fresh directory for history frames and the run receipt")
    parser.add_argument("--nest-triggers", type=Path, default=None, metavar="TOML",
                        help="arm a [hex_nests] table: the loop looks on the card every cadence_s and "
                             "writes each firing to <out>/nest-triggers.jsonl for `add-nest --watch`")
    parser.add_argument("--stochastic", default=None, metavar="JSON",
                        help="per-member SPPT/SKEB configuration, default off; see stochastic.py")
    parser.add_argument("--members", type=Path, default=None, metavar="JSON",
                        help="opt-in roster of 1 to 4 resident point-cull members on one card")
    parser.add_argument("--member-buffers", choices=("auto", "nvme"), default="auto",
                        help="shared read-only tables: checked shm with NVMe fallback, or NVMe")
    parser.add_argument(
        "--scratch", type=Path, default=None, metavar="DIR",
        help="kernel-cache dir OUTSIDE --out "
             "(default: sibling <out>.forecast-scratch)")
    parser.add_argument(
        "--receipt", type=Path, default=None, metavar="FILE",
        help="run receipt path (default: <out>/forecast-receipt.json)")
    parser.add_argument(
        "--gpuwm-checkout", type=Path, default=None, metavar="DIR",
        help="woof tree whose seam files run (default: the installed "
             "engine, recorded by version and pip's RECORD digest); a git "
             "clone at the pinned tag is recorded by commit instead")
    parser.add_argument(
        "--repo", type=Path, default=None, metavar="DIR",
        help="accepted for older scripts and not used: the forecast drivers "
             "ship in the package since 0.3.2")
    parser.add_argument(
        "--case-label", default=None, metavar="TEXT",
        help="label recorded in the receipt")
    parser.add_argument(
        "--horiz-mixing", choices=("2d_smagorinsky", "off"),
        default="2d_smagorinsky",
        help="horizontal mixing lane; the default is the native Registry "
             "deformation-based 2-D Smagorinsky. 'off' is the pre-mixing "
             "control lane, reported as the configuration native itself "
             "cannot integrate on convective cases (default: 2d_smagorinsky)")
    parser.add_argument(
        "--convection", default="auto", metavar="SCHEME",
        help="WOOF-route cumulus selection (a scheme other than auto selects the "
             "WOOF route when no suite is named): auto, off, gf, ntiedtke, or any cumulus "
             "option of WOOF's physics registry by MPAS name, registry option "
             "id or cu_physics number (kain-fritsch, cu_tiedtke, 6). "
             "The default 'auto' switches the cumulus "
             "scheme off where the bound mesh's finest spacing is below 3 km, "
             "with no flag passed: Grell-Freitas is called every model step, "
             "so a fine mesh's short step calls it up to 24 times as often as "
             "the proven 120 s, which measured as a different solution "
             "(vertical velocity still climbing past 80 m/s at 2 h), and "
             "convection is resolved explicitly at those spacings. 'off' and "
             "'gf' are explicit A/B arms and record themselves as explicit; "
             "'ntiedtke' selects New Tiedtke, which is also what 'auto' "
             "picks at 20 km and coarser (default: auto)")
    parser.add_argument(
        "--preset", choices=forecast_preset.preset_names(),
        default=forecast_preset.DEFAULT_PRESET,
        help="named forecast preset (woof.hex.forecast_preset). "
             + " ".join(
                 f"'{row.name}': {row.summary}."
                 for row in (forecast_preset.PRESETS[name]
                             for name in forecast_preset.preset_names())
             )
             + " A preset sets --pbl-cadence at the mesh's timestep; an "
             "explicit --pbl-cadence wins "
             f"(default: {forecast_preset.DEFAULT_PRESET})")
    parser.add_argument(
        "--pbl-cadence", default=None, metavar="{auto,SECONDS}",
        help="surface/PBL cadence in seconds (config_bldt_seconds); when "
             "absent the --preset row decides it. 'auto' is the PROVEN "
             "CONFIGURATION: welded to "
             "config_dt, so the surface layer, the land-surface model and "
             "the PBL run every model step, as the native x4 v8.4.1 "
             "reference ran. A number of seconds calls the stack once every "
             "SECONDS/dt steps and holds its tendency on the steps between "
             "(the engine's own positive-bldt path); it must be a whole "
             "number of steps or it is refused by name, it changes the "
             "forecast, and it records itself as an explicit selection with "
             "its calls per hour in the receipt. Radiation keeps its own "
             "600 s cadence either way. Measured on the point mesh in "
             "docs/hex-point-hrrr.md (default: the preset's)")
    parser.add_argument(
        "--local-timestep", action="store_true",
        help="OPT-IN, default off: advance coarse columns on fewer, longer "
             "acoustic sub-steps. Native v8.4.1 has none, so this is a "
             "declared divergence and does not pay on the published "
             "variable-resolution mesh (0.988x measured). On a limited-area "
             "cull every driven boundary cell is held at rate 1, and on the "
             "culls mesh-plan makes that leaves one class: the run is then "
             "bit-identical to the default (measured 2026-09-14)")
    parser.add_argument(
        "--local-timestep-rates", default="1,3", metavar="LADDER",
        help="acoustic rate ladder; strictly increasing, starts at 1 "
             "(default: 1,3)")
    parser.add_argument(
        "--local-timestep-buffer-rings", type=int, default=1, metavar="N",
        help="rings demoted to the finer rate at a class boundary (default: 1)")
    parser.add_argument(
        "--local-timestep-classing", default=None, metavar="NPZ",
        help="A/B INSTRUMENT, not a feature: per-cell acoustic rates from a "
             ".npz (key cell_rate) instead of the grid file's dcEdge, so a "
             "class interface can be placed in a cull's interior and measured. "
             "Needs --local-timestep. tools/lts_forced_classing.py writes it")
    parser.add_argument(
        "--headroom-mib", type=float, default=None, metavar="MIB",
        help="override the margin the admission decision holds back from the "
             "card.  The default is the model's own margin, priced from this "
             "card: the RRTMG shortwave workspace (1,745.6 MiB on a 170 SM "
             "part, 872.8 MiB on a 68 or 70 SM part) plus 11.2 MiB of "
             "instrument convention.  The retired flat 512 MiB is still "
             "computable at device_admission.RETIRED_FLAT_HEADROOM_BYTES; it "
             "named no breakage and failed by 96 MiB on a real run")
    parser.add_argument(
        "--device-fixed-mib", type=float, default=None, metavar="MIB",
        help="this card's measured fixed footprint term; supply with "
             "--device-bytes-per-cell to replace the shipped row")
    parser.add_argument(
        "--device-bytes-per-cell", type=float, default=None, metavar="BYTES",
        help="this card's measured per-cell slope; supply with "
             "--device-fixed-mib")
    parser.add_argument(
        "--checkpoint-hours", default=None, metavar="H[,H...]",
        help="forecast hours (each a history frame) at which to write a restart "
             "checkpoint; the driver resolves them")
    parser.add_argument(
        "--restart-from", type=Path, default=None, metavar="PKL",
        help="resume from a checkpoint this run wrote; --start-time/--hours "
             "name the ORIGINAL run")
    parser.add_argument(
        "--da-analysis", type=Path, default=None, metavar="NC",
        help="DA-cycling restart: with --restart-from, replace the dycore "
             "state by this analysis; physics resumes from the checkpoint")
    parser.add_argument(
        "--jedi-background", action="store_true",
        help="at every history frame also write the JEDI background file")
    parser.add_argument(
        "--stop-on-refusal", action="store_true",
        help="when the model refuses to publish a step, stop and write the "
             "receipt for the frames already committed instead of aborting "
             "with none. No validation is relaxed")
    parser.add_argument(
        "--iau-analysis", type=Path, default=None, metavar="INIT",
        help="incremental analysis update: --init is the background and this file the analysis "
             "(same mesh and cell order); their difference enters over --iau-window-seconds instead "
             "of being inserted at the start, by --iau-route")
    parser.add_argument(
        "--iau-window-seconds", type=float, default=None, metavar="S",
        help="IAU window length from the start of the run")
    parser.add_argument(
        "--balance-series", action="store_true",
        help="write balance-series.jsonl: surface-pressure tendency statistics at every macro "
             "boundary (the initialization-shock measure)")
    from .analysis_door import add_analysis_arguments

    add_analysis_arguments(parser)
    parser.add_argument(
        "--preflight", action="store_true",
        help="resolve, bind and admit without touching the integration")
    parser.add_argument(
        "--no-products", action="store_true",
        help="end at the history frames instead of drawing the standard "
             "product set (maps on every view the mesh has, soundings, "
             "meteograms) into <out>/products; prints the products command "
             "instead")
    parser.add_argument(
        "--products-detached", action=argparse.BooleanOptionalAction, default=True,
        help="the product stage runs as its own CPU-only process (log: "
             "<out>/products.log) and this command returns as soon as the "
             "receipt is written, so the cards are released while the maps "
             "draw (the default). --no-products-detached runs it inside this "
             "command instead, which keeps this process -- and every card it "
             "holds -- alive until the products are on disk")
    from .products_door import add_label_arguments

    add_label_arguments(parser)


def add_forecast_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "forecast",
        help="run the model on a registered mesh from an init "
             "(GPU)",
        description=(
            "Drive the MPAS-A v8.4.2 CUDA forecast on a registered mesh. The door "
            "binds the mesh against the registry's pinned bytes, admits the "
            "request against device memory measured at that moment, runs the "
            "integration, and writes history frames the render door consumes "
            "plus one run receipt."
        ),
    )
    add_forecast_arguments(parser)
    parser.set_defaults(handler=run_forecast)
