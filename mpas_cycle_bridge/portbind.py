"""The port-facing half of the seam, with no woof in it.

This is the code that loads the MPAS GPU port by path, verifies every
symbol it will call, maps an anchor's array mappings onto the port's
state objects, and takes a boundary off the device.  It lives in the
bridge rather than in woof.cycle for one reason: the forecast
worker runs in a process where importing woof is forbidden until the
port itself pins its frozen Arwen checkout, so anything the worker calls
must be reachable without the woof name.

woof.cycle.mpas_port_adapter re-exports everything here and adds
the spine-side refusal type.  There is one implementation; both sides
run it.

"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


#: Relative paths of the port tools this seam loads, in load order.  The
#: forecast tool loads the proof itself and re-pins several of its
#: module-level authorities against the init file, so it must be loaded
#: through its own module rather than reimplemented.
PORT_PROOF_RELPATH = "tools/run_cuda_v841_full_physics_x4.py"
PORT_FORECAST_RELPATH = "tools/run_cuda_v841_forecast.py"
PORT_MESH_BINDING_RELPATH = "tools/mpas_mesh_binding.py"
PORT_SRC_RELDIR = "src"

#: The port's Python package, one row per source layout the port has
#: shipped, newest first.  The port renamed ``mpas_port`` to ``woof.hex``
#: in its 0.2.0 package rename (2026-08-28); its tools import whichever
#: one the tree holds.  A bound tree's package is the first row whose
#: ``src/<package>/__init__.py`` exists (:func:`port_package`), and every
#: port import in this seam goes through it (:meth:`PortBinding.module`).
#: The spine prices a leg with that package's ``device_admission`` module,
#: found by the same rows in the same order
#: (``woof.cycle.mpas_bridge.PARENT_DEVICE_FOOTPRINT``).
PORT_PACKAGES = ("hexcore", "mpas_port")

#: Every symbol this seam calls on the proof module.  Checked up front so
#: a binding failure names the whole list rather than dying on the first
#: attribute access halfway through a GPU run.
REQUIRED_PROOF_SYMBOLS = (
    "_construct_device_stack",
    "_run_steps",
    "fingerprint_execution_boundary",
    "require_frozen_execution_sources",
    "require_fingerprint_identity",
    "DT_SECONDS",
)

REQUIRED_FORECAST_SYMBOLS = (
    "prepare_forecast_host",
    "verify_forecast_authorities",
    "proof",
)

#: The prognostic fields that cross the anchor boundary, in the order
#: the port's ``state.PrognosticState`` declares them.
PROGNOSTIC_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")

#: The saved-diagnostics sidecar.  MPAS carries ``theta_m`` and ``exner``
#: independently of the mass products; rebuilding either from
#: ``rho_theta`` loses float32 source-order bits, which is why the port
#: threads this sidecar through every step and why an anchor must carry
#: it rather than recompute it.
SAVED_DIAGNOSTIC_FIELDS = ("theta_m", "exner", "density_perturbation",
                           "rho_theta_perturbation", "pressure_perturbation",
                           "normal_velocity", "vertical_velocity")

#: The vertical-grid metric a boundary must carry beside its state.
#:
#: MPAS's prognostic mass variable is ``rho_zz`` and its equation of
#: state is ``exner = (zz*(rd/p0)*rho_theta) ** (rd/cv)``
#: (the port's ``cuda_backend.recovery::pressure_point``).  ``zz`` is
#: therefore not decoration on the boundary: without it nothing outside
#: the port can recompute exner, and the spine's consistency instrument
#: spent a whole closed-loop proof reading 0.407 on a correct state for
#: exactly that reason.  It is taken from the SAME device array the
#: recovery kernel was handed, not rebuilt from the init file, so the
#: grader and the producer cannot disagree about the mesh.
VERTICAL_METRIC_FIELDS = ("zz",)

#: The exact key set ``PersistentTwoPhaseCudaPhysicsBackendV841
#: .restore_restart_state`` demands.  It compares by ``set(payload) !=
#: expected_keys``, so a round trip that adds or drops one key is
#: refused by the port -- which is the behaviour we want.
BACKEND_RESTART_KEYS = frozenset({"schema", "identity", "seam", "adapter"})


class PortBindingError(RuntimeError):
    """The port could not be bound, and says every place it looked.

    Mirrors woof.cycle.contracts.CycleRefusal's discipline -- a
    refusal must name what it observed -- without importing it, because
    this module runs in processes where woof is not importable.  The
    spine-side adapter re-raises these as CycleRefusal subclasses so
    the spine's fail-closed handling still sees one type.
    """

    def __init__(self, message: str, **observed: object) -> None:
        if not observed:
            raise ValueError(
                "a port binding refusal must name what was observed; "
                f"got bare message {message!r}")
        rendered = ", ".join(f"{key}={observed[key]!r}"
                             for key in sorted(observed))
        super().__init__(f"{message} | observed: {rendered}")
        self.observed = dict(observed)


@dataclass(frozen=True)
class PortBinding:
    """A loaded, symbol-verified port tree."""

    root: Path
    proof: Any
    forecast: Any
    mesh_binding: Any
    #: The row of :data:`PORT_PACKAGES` this tree holds.
    package: str
    source_receipt: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dt_seconds(self) -> float:
        return float(self.proof.DT_SECONDS)

    def module(self, name: str) -> Any:
        """The port module ``name`` (``state``, ``driver``, ...) from the
        package this tree holds, so no call site spells a package name."""
        return importlib.import_module(f"{self.package}.{name}")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise PortBindingError("port tool is not loadable as a module",
                               module=name, path=str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def port_package(port_root: str | Path) -> str | None:
    """The first row of :data:`PORT_PACKAGES` the tree holds, or ``None``."""
    src = Path(port_root).expanduser() / PORT_SRC_RELDIR
    for package in PORT_PACKAGES:
        if (src / package / "__init__.py").is_file():
            return package
    return None


def bind_port(port_root: str | Path, *,
              verify_frozen_sources: bool = True) -> PortBinding:
    """Load the port tools by path and verify every symbol we will call.

    ``verify_frozen_sources`` runs the port's own
    ``require_frozen_execution_sources()`` -- the six-module SHA-256
    freeze check -- before anything else touches a device.  It is on by
    default because a cycle that integrated an unfrozen dycore is not
    the cycle anyone asked for.
    """
    root = Path(port_root).expanduser()
    looked_in = [str(root / PORT_SRC_RELDIR), str(root / PORT_PROOF_RELPATH),
                 str(root / PORT_FORECAST_RELPATH),
                 str(root / PORT_MESH_BINDING_RELPATH)]
    if not root.is_dir():
        raise PortBindingError("port root does not exist",
                               port_root=str(root), looked_in=looked_in)
    missing_files = [candidate for candidate in
                     (PORT_SRC_RELDIR, PORT_PROOF_RELPATH,
                      PORT_FORECAST_RELPATH, PORT_MESH_BINDING_RELPATH)
                     if not (root / candidate).exists()]
    packages = [f"{PORT_SRC_RELDIR}/{name}" for name in PORT_PACKAGES]
    if missing_files:
        raise PortBindingError(
            "port root is missing members this seam binds to",
            port_root=str(root), missing=missing_files, looked_in=looked_in,
            remedy=f"point --port-root at the tree that holds one of "
                   f"{', '.join(packages)} and tools/run_cuda_v841_*.py, "
                   "not at its parent")

    # Refused here, in the worker, before any tool loads: a tree with
    # neither package passes the member check above, and the worker then
    # stopped at its first port import with "No module named ..." after
    # the spine had already priced the leg.
    package = port_package(root)
    if package is None:
        raise PortBindingError(
            "the port tree holds none of the port packages this seam "
            "imports", port_root=str(root), looked_for=packages,
            remedy="point --port-root at a port checkout whose src/ holds "
                   "one of those packages; a port in a new layout is one "
                   "more row of mpas_cycle_bridge.portbind.PORT_PACKAGES")

    src = str(root / PORT_SRC_RELDIR)
    if src not in sys.path:
        sys.path.insert(0, src)

    forecast = _load_module("v841_forecast", root / PORT_FORECAST_RELPATH)
    proof = getattr(forecast, "proof", None)
    if proof is None:
        proof = _load_module("v841_proof", root / PORT_PROOF_RELPATH)
    mesh_binding = _load_module("mpas_mesh_binding",
                                root / PORT_MESH_BINDING_RELPATH)

    absent = [name for name in REQUIRED_PROOF_SYMBOLS
              if not hasattr(proof, name)]
    absent += [f"forecast.{name}" for name in REQUIRED_FORECAST_SYMBOLS
               if not hasattr(forecast, name)]
    if absent:
        raise PortBindingError(
            "the port tree does not carry the symbols this seam calls",
            port_root=str(root), missing_symbols=absent,
            proof_module=str(root / PORT_PROOF_RELPATH),
            forecast_module=str(root / PORT_FORECAST_RELPATH),
            looked_in=looked_in,
            remedy="this seam is written against mesh binding at 12fe6a3; "
                   "a port tree that lacks these is a different vintage and "
                   "must not be bound silently")

    receipt: Mapping[str, Any] = {}
    if verify_frozen_sources:
        receipt = dict(proof.require_frozen_execution_sources())
    return PortBinding(root=root, proof=proof, forecast=forecast,
                       mesh_binding=mesh_binding, package=package,
                       source_receipt=receipt)


# --------------------------------------------------------------------------
# host state <-> anchor mapping


def prognostic_state(binding: PortBinding,
                     mapping: Mapping[str, np.ndarray]) -> Any:
    """Build the port's :class:`PrognosticState` from an anchor mapping."""
    PrognosticState = binding.module("state").PrognosticState

    missing = [name for name in PROGNOSTIC_FIELDS if name not in mapping]
    if missing:
        raise PortBindingError("anchor prognostics cannot make a port state",
                               missing=missing,
                               present=sorted(mapping))
    time_seconds = float(np.asarray(
        mapping.get("time_seconds", 0.0)).reshape(-1)[0])
    return PrognosticState(
        **{name: np.ascontiguousarray(mapping[name])
           for name in PROGNOSTIC_FIELDS},
        time_seconds=time_seconds)


def saved_diagnostics(binding: PortBinding,
                      mapping: Mapping[str, np.ndarray]) -> Any:
    """Build :class:`DrySavedDiagnostics` from an anchor's derived block."""
    DrySavedDiagnostics = binding.module("driver").DrySavedDiagnostics

    missing = [name for name in SAVED_DIAGNOSTIC_FIELDS
               if name not in mapping]
    if missing:
        raise PortBindingError(
            "anchor derived block cannot make the port's saved-diagnostics "
            "sidecar; the dycore cannot resume without every one of them",
            missing=missing, present=sorted(mapping),
            required=list(SAVED_DIAGNOSTIC_FIELDS))
    return DrySavedDiagnostics(
        **{name: np.ascontiguousarray(mapping[name])
           for name in SAVED_DIAGNOSTIC_FIELDS})


# --------------------------------------------------------------------------
# the physics backend's restart payload, split for the anchor


def split_backend_restart(payload: Mapping[str, Any]
                          ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Split the backend restart payload into JSON skeleton + arrays.

    The payload is ``{schema, identity, seam, adapter}``; every array in
    it lives under ``seam["arrays"]`` and everything else is JSON-able.
    So the anchor stores the arrays in ``parent_seam.nc`` -- the slot the
    format already reserves for "the physics-seam mapping" -- and the
    skeleton in the manifest.  No pickle crosses the boundary, which is
    the whole reason the anchor format exists.
    """
    if set(payload) != set(BACKEND_RESTART_KEYS):
        raise PortBindingError(
            "backend restart payload does not carry the port's exact keys",
            observed=sorted(payload), expected=sorted(BACKEND_RESTART_KEYS))
    seam = payload["seam"]
    arrays = {str(name): np.ascontiguousarray(value)
              for name, value in dict(seam["arrays"]).items()}
    skeleton = {key: value for key, value in payload.items() if key != "seam"}
    skeleton["seam"] = {key: value for key, value in dict(seam).items()
                        if key != "arrays"}
    skeleton["seam"]["array_names"] = sorted(arrays)
    return skeleton, arrays


def join_backend_restart(skeleton: Mapping[str, Any],
                         arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Rebuild the exact payload ``restore_restart_state`` demands."""
    seam_skeleton = dict(skeleton.get("seam") or {})
    names = list(seam_skeleton.pop("array_names", ()))
    unexpected = sorted(set(arrays) - set(names))
    absent = sorted(set(names) - set(arrays))
    if unexpected or absent:
        raise PortBindingError(
            "seam arrays on disk do not match the names the manifest "
            "recorded; the anchor's seam block was edited or truncated",
            missing=absent, unexpected=unexpected)
    seam_skeleton["arrays"] = {name: np.ascontiguousarray(arrays[name])
                               for name in names}
    payload = {key: value for key, value in skeleton.items() if key != "seam"}
    payload["seam"] = seam_skeleton
    if set(payload) != set(BACKEND_RESTART_KEYS):
        raise PortBindingError(
            "rebuilt backend restart payload has the wrong key set",
            observed=sorted(payload), expected=sorted(BACKEND_RESTART_KEYS))
    return payload


# --------------------------------------------------------------------------
# the download the port does not export


def download_stack(binding: PortBinding, stack: Mapping[str, Any], *,
                   label: str) -> dict[str, Any]:
    """Take a committed boundary off the device, checked on the way.

    This is the ``_download_state`` the first adapter reached for.  It
    does what ``download_driver_checkpoint`` does -- synchronise, pull
    the atmosphere state and the saved sidecar to the host, take the
    backend's restart payload -- and it keeps that function's
    device-to-host fingerprint identity check, which is the no-ECC guard
    on the transfer itself.  What it drops is the hard assertion that
    the clock reads exactly F030: a cycle boundary lands where the cycle
    clock puts it.
    """
    from types import SimpleNamespace

    fingerprint_atmosphere = binding.module(
        "cuda_dualrun").fingerprint_atmosphere
    DrySavedDiagnostics = binding.module("driver").DrySavedDiagnostics

    driver = stack["driver"]
    backend = stack["backend"]
    cp = driver.cp
    cp.cuda.get_current_stream().synchronize()

    device_fingerprint = fingerprint_atmosphere(driver.atmosphere)
    state = driver.atmosphere.state.to_host()
    saved = driver.atmosphere.saved
    host_saved = DrySavedDiagnostics(
        **{name: cp.asnumpy(getattr(saved, name))
           for name in SAVED_DIAGNOSTIC_FIELDS})
    cp.cuda.get_current_stream().synchronize()
    host_fingerprint = fingerprint_atmosphere(
        SimpleNamespace(state=state, saved=host_saved))
    binding.proof.require_fingerprint_identity(
        f"{label} device-to-host boundary download",
        device_fingerprint, host_fingerprint)

    prognostic = {name: np.ascontiguousarray(getattr(state, name))
                  for name in PROGNOSTIC_FIELDS}
    prognostic["time_seconds"] = np.asarray(float(state.time_seconds))
    derived = {name: np.ascontiguousarray(getattr(host_saved, name))
               for name in SAVED_DIAGNOSTIC_FIELDS}
    metric = {name: np.ascontiguousarray(cp.asnumpy(
        getattr(driver.atmosphere.vertical, name)))
        for name in VERTICAL_METRIC_FIELDS}
    return {
        "prognostic": prognostic,
        "derived": derived,
        "metric": metric,
        "backend_restart": backend.restart_state(),
        "model_time_seconds": float(state.time_seconds),
        "atmosphere_fingerprint": device_fingerprint,
    }
