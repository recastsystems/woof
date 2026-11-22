"""The spine's side of the bridge: launch a forecast, read a segment.

The cycling spine cannot run the MPAS GPU port in its own process.  The
port pins woof by commit and loads it from a frozen Arwen checkout, and
refuses if woof is already live from another tree -- and ``woof.cycle``
IS woof.  So the forecast runs in a process of its own, launched from
here, and **the anchor on disk is the only channel between them**.

The channel, stated exactly, because "the anchor is the channel" is the
kind of sentence that quietly stops being true:

* IN: the worker is handed one committed anchor directory.  It reads the
  parent prognostics, the saved-diagnostics sidecar, the physics
  backend's restart payload from the anchor's seam slot, and the
  analysis increment -- all of it from that directory, all of it in the
  anchor's own array format.  Nothing else about the previous cycle
  reaches it.
* OUT: the worker writes a *segment* -- the same array format, plus a
  JSON receipt of what it did.  The spine reads that, grades the
  evidence, and seals it into the next anchor with :func:`write_anchor`,
  which is where the refusals and the parent-kind stamp live.
* Never: a pickle, a shared object, a live import.

A segment is deliberately not an anchor.  The stamp on an anchor is a
claim about evidence (``mpas-cuda`` means the analysis genuinely
re-entered the dycore), and the process that produced the evidence
should not be the process that grades it.  :func:`stamp_for_segment`
grades it, on this side.

The import constraint is enforced structurally in four places, not by
convention:

1. ``mpas_cycle_bridge`` contains no woof import anywhere, and
   :func:`verify_bridge_purity` proves that by parsing the package's own
   syntax trees.  A test calls it; a contributor who adds one turns that
   test red.
2. The worker installs an import guard on ``sys.meta_path`` that refuses
   ``woof.cycle`` by name, always, and ``woof`` itself until the
   port's pinned checkout has been declared.
3. :func:`launch` builds the child environment with ``PYTHONSAFEPATH=1``
   and strips ``PYTHONPATH``, so the spine tree is not on the child's
   import path at all; the worker adds the repository root only long
   enough to import the bridge, then removes it again.
4. The worker records where ``woof`` actually came from in its segment
   receipt, so every run carries the evidence that the constraint held
   rather than the assertion that it does.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import mpas_cycle_bridge
from mpas_cycle_bridge.portbind import PORT_PACKAGES, PORT_SRC_RELDIR

from woof.cycle.contracts import CycleRefusal

WORKER_RELPATH = Path("mpas_cycle_bridge") / "worker.py"
SEGMENT_MANIFEST = "segment.json"
WORKER_STATUS = "worker-status.json"
WORKER_LOG = "worker.log"

#: Environment variables the child keeps.  Everything else is dropped:
#: a forecast that only reproduces under somebody's shell profile is not
#: reproducible.  ``PYTHONPATH`` is deliberately absent.
ENV_PASSTHROUGH = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TMPDIR",
    "TEMP", "TMP", "SystemRoot", "COMSPEC", "windir", "USERPROFILE",
    "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "CUPY_CACHE_DIR", "NVIDIA_VISIBLE_DEVICES", "CUDA_CACHE_PATH",
    # Windows resolves the per-user site-packages directory through
    # APPDATA.  Drop it and the child loses numpy -- which looks exactly
    # like a broken bridge and is not one.  VIRTUAL_ENV is kept for the
    # same reason on either platform.
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PATHEXT",
    "SYSTEMDRIVE", "NUMBER_OF_PROCESSORS", "VIRTUAL_ENV", "CONDA_PREFIX",
)


class BridgeRefusal(CycleRefusal):
    """The bridge refused to run or to believe a forecast process."""


def bridge_root() -> Path:
    """The repository root that holds both ``woof`` and the bridge."""
    return Path(mpas_cycle_bridge.__file__).resolve().parent.parent


def worker_path() -> Path:
    return bridge_root() / WORKER_RELPATH


# --------------------------------------------------------------------------
# rule 1: the bridge package carries no woof import, provably


def verify_bridge_purity() -> dict[str, Any]:
    """Parse every bridge module and refuse if any of them imports woof.

    This is the constraint made mechanical.  ``mpas_cycle_bridge`` is
    the code the forecast process runs; the moment one of its modules
    imports woof, the forecast process claims the package name and the
    port refuses to pin its frozen Arwen -- eight frames deep, with a
    message about Arwen rather than about the import that caused it.  So
    the import is caught here instead, by reading the source.

    Docstrings may say "woof" as much as they like.  Only ``import``
    statements count.
    """
    package = Path(mpas_cycle_bridge.__file__).resolve().parent
    offences: list[dict[str, Any]] = []
    scanned: list[str] = []
    for module in sorted(package.rglob("*.py")):
        if "__pycache__" in module.parts:
            continue
        scanned.append(module.name)
        tree = ast.parse(module.read_text(encoding="utf-8"),
                         filename=str(module))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name == "woof" or name.startswith("woof."):
                    offences.append({"module": module.name,
                                     "line": node.lineno, "imports": name})
    if offences:
        raise BridgeRefusal(
            "the bridge imports woof, which breaks the whole reason it "
            "exists: the forecast process must let the port's pinned WOOF-ArWen "
            "checkout be the first tree to claim the woof name",
            offences=offences, package=str(package),
            remedy="move whatever this needed into mpas_cycle_bridge, or "
                   "pass it across the anchor on disk")
    return {"modules_scanned": scanned, "gpuwm_imports": 0}


# --------------------------------------------------------------------------
# rule 3: a child environment the spine tree is not reachable from


def child_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {name: os.environ[name] for name in ENV_PASSTHROUGH
           if name in os.environ}
    # PYTHONSAFEPATH stops Python prepending the script's directory, and
    # PYTHONPATH is never forwarded.  Between them the child starts with
    # no path entry that can resolve ``woof.cycle``; the worker adds the
    # repository root itself, only for as long as it takes to import the
    # bridge, and removes it before the port pins anything.
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


# --------------------------------------------------------------------------
# launching


def launch(*, phase: str, port_root: str | os.PathLike[str],
           port_config: str | os.PathLike[str], steps: int,
           out: str | os.PathLike[str],
           anchor: str | os.PathLike[str] | None = None,
           python: str | None = None,
           env: Mapping[str, str] | None = None,
           timeout: float | None = None,
           check: bool = True) -> dict[str, Any]:
    """Run one forecast segment in a fresh process and return its receipt.

    Returns the parsed ``segment.json``.  Refuses -- naming the child's
    exit code, its log and its recorded traceback -- if the child failed,
    if it wrote no segment, or if it reports that woof came from
    anywhere but the pinned Arwen checkout.
    """
    out_dir = Path(out).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_bridge_purity()

    argv: list[str] = [python or sys.executable, str(worker_path()), phase,
                       "--port-root", str(port_root),
                       "--port-config", str(port_config),
                       "--steps", str(int(steps)),
                       "--out", str(out_dir)]
    if anchor is not None:
        argv += ["--anchor", str(anchor)]

    log_path = out_dir / WORKER_LOG
    completed = subprocess.run(argv, env=child_environment(env),
                               cwd=str(bridge_root()), capture_output=True,
                               text=True, timeout=timeout)
    log_path.write_text(
        f"$ {' '.join(argv)}\n\n--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}\n", encoding="utf-8")

    status_path = out_dir / WORKER_STATUS
    status = (json.loads(status_path.read_text(encoding="utf-8"))
              if status_path.is_file() else None)
    if completed.returncode != 0 or (status or {}).get("ok") is not True:
        if not check:
            return {"ok": False, "returncode": completed.returncode,
                    "log": str(log_path), "status": status}
        raise BridgeRefusal(
            "the MPAS forecast worker did not complete",
            phase=phase, returncode=completed.returncode,
            log=str(log_path),
            worker_error=(status or {}).get("error", "<no status file>"),
            stderr_tail=completed.stderr[-2000:] or "<empty>")

    manifest_path = _segment_manifest_path(out_dir, phase)
    if not manifest_path.is_file():
        raise BridgeRefusal(
            "the forecast worker exited zero but published no segment",
            phase=phase, expected=str(manifest_path), log=str(log_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pin = manifest.get("arwen_pin") or {}
    if pin.get("obeyed") is not True:
        raise BridgeRefusal(
            "the forecast process did not load woof from the port's pinned "
            "WOOF-ArWen checkout, so its numbers are not the pinned physics",
            phase=phase, arwen_pin=pin, log=str(log_path))
    if pin.get("spine_modules"):
        raise BridgeRefusal(
            "the cycling spine was live inside the forecast process",
            phase=phase, spine_modules=pin["spine_modules"],
            log=str(log_path))
    manifest["_dir"] = str(manifest_path.parent)
    return manifest


# --------------------------------------------------------------------------
# sizing: does the port's device stack for this mesh fit this card?


#: How a model parent's device footprint is priced, one row per parent
#: kind, so a second engine is a row here rather than a branch.
#:
#: The row names WHERE the port keeps its measured capacity model, not a
#: copy of the model's numbers.  The port states the model in one module,
#: ``device_admission``, and every one of its own admission decisions
#: answers from it, because two copies of one footprint model are what let
#: a card between their answers pass one gate and die on the other.  That
#: module is stdlib-only by the port's own contract, so it is read by path
#: here without importing the port.  Its surface is
#: ``card_profile_from_attributes``, ``model_for_card`` and
#: ``required_free_bytes``.
#:
#: The 2026-08-20 affine fit (5,018 MiB fixed plus 140,916 B per cell, on
#: the x1.40962 and x4.163842 anchors) is NOT held here: the port retired
#: that shape on 2026-08-27 for a card-scaled core, tiled physics
#: workspaces and a margin of two named terms, and priced with it a
#: 32 GiB card would be refused the port's registered 224,210-cell meshes,
#: which the port's own model admits.
PARENT_DEVICE_FOOTPRINT: Mapping[str, Mapping[str, Any]] = {
    "mpas-cuda": {
        # One path per port layout, from the bridge's own package rows
        # in their order; the first that exists is the model.  The worker
        # imports the port through the same rows, so a leg is priced and
        # run from one package.
        "admission_module": tuple(
            f"{PORT_SRC_RELDIR}/{package}/device_admission.py"
            for package in PORT_PACKAGES),
        # The device stack the bridge builds: the global mesh's.
        "configuration": "global",
        # The port config's mesh file and the dimension that counts cells.
        "mesh_file_role": "grid",
        "cells_dimension": "nCells",
        # The file whose header carries the vertical level count the
        # port's tiled workspaces scale with, and that dimension.
        "levels_file_role": "init",
        "levels_dimension": "nVertLevels",
    },
}

#: The functions a port's admission module must publish to price a leg.
#: Port trees from 2026-08-25 to 2026-08-27 have a ``device_admission``
#: with ``required_free_bytes`` only, on the retired affine model.
ADMISSION_SURFACE = ("card_profile_from_attributes", "model_for_card",
                     "required_free_bytes")

#: Printed by the card probe: the two numbers the port's model reads off
#: the card, and the free memory it is compared with, from a process of
#: its own so this one never holds a CUDA context the leg then cannot use.
_CARD_PROBE = r"""
import json
import cupy as cp
device = cp.cuda.Device()
attributes = device.attributes
free, total = cp.cuda.runtime.memGetInfo()
name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
print(json.dumps({
    "name": name.decode(errors="replace") if isinstance(name, bytes) else str(name),
    "MultiProcessorCount": int(attributes["MultiProcessorCount"]),
    "MaxThreadsPerMultiProcessor": int(attributes["MaxThreadsPerMultiProcessor"]),
    "free_bytes": int(free), "total_bytes": int(total)}))
"""


def _port_admission(port_root: Path, row: Mapping[str, Any]):
    import importlib.util

    for relative in row["admission_module"]:
        path = port_root / relative
        if path.is_file():
            # Registered before it runs: the port's dataclasses resolve
            # their string annotations through sys.modules at class
            # creation, and an unregistered module fails right there.
            name = "_mpas_port_device_admission"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            return module, path
    return None, None


def probe_card(python: str | None = None) -> dict[str, Any]:
    """The card the leg will run on, read by a short-lived process."""

    completed = subprocess.run([python or sys.executable, "-c", _CARD_PROBE],
                               env=child_environment(), capture_output=True,
                               text=True, timeout=120)
    if completed.returncode != 0:
        raise BridgeRefusal(
            "the card the forecast worker would use could not be read, so "
            "the leg cannot be sized before it builds the device stack",
            returncode=completed.returncode,
            stderr_tail=completed.stderr[-2000:] or "<empty>",
            remedy="woof doctor names why the card is unreachable")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _port_file(config: Mapping[str, Any], role: str) -> Path:
    """A file the port config names, found where the worker finds it.

    The worker runs with the bridge root as its working directory
    (:func:`launch`), so a relative path in the port config is relative
    to that root and not to wherever the spine was started.
    """

    path = Path(config[role]).expanduser()
    return path if path.is_absolute() else bridge_root() / path


def mesh_cells(port_config: str | os.PathLike[str], *,
               parent_kind: str = "mpas-cuda") -> tuple[int, str]:
    """``(cells, mesh name)`` of the mesh a port config binds, off its header."""

    from woof.netcdf_bridge import open_dataset

    row = PARENT_DEVICE_FOOTPRINT[parent_kind]
    config = json.loads(Path(port_config).expanduser().read_text(encoding="utf-8"))
    grid = _port_file(config, row["mesh_file_role"])
    dimensions = open_dataset(grid).dimensions
    if row["cells_dimension"] not in dimensions:
        raise BridgeRefusal(
            "the port config's mesh file has no cell dimension, so the leg "
            "cannot be sized", mesh_file=str(grid),
            dimension=row["cells_dimension"], have=sorted(dimensions))
    return (int(dimensions[row["cells_dimension"]].size),
            str(config.get("mesh", grid.name)))


def mesh_levels(port_config: str | os.PathLike[str], *,
                parent_kind: str = "mpas-cuda") -> int | None:
    """The vertical level count of the case a port config binds.

    Read off the header of the file the row names; ``None`` when that
    file carries no such dimension, and the leg is then priced on the
    port's own default level count, which the receipt says.
    """

    from woof.netcdf_bridge import open_dataset

    row = PARENT_DEVICE_FOOTPRINT[parent_kind]
    config = json.loads(Path(port_config).expanduser().read_text(encoding="utf-8"))
    dimensions = open_dataset(
        _port_file(config, row["levels_file_role"])).dimensions
    if row["levels_dimension"] not in dimensions:
        return None
    return int(dimensions[row["levels_dimension"]].size)


def admit_parent_device(*, port_root: str | os.PathLike[str], cells: int,
                        mesh: str, card: Mapping[str, Any],
                        levels: int | None = None,
                        parent_kind: str = "mpas-cuda") -> dict[str, Any]:
    """Refuse, before a leg starts, a mesh the card's free memory cannot hold.

    The concrete breakage: the worker builds the mesh's host state and only
    then constructs the port's device stack, and nothing between the two
    asked whether the card could hold it, so a mesh too big for the card
    died inside the worker's first CuPy allocation, after the host
    preparation.  The price is the port's own
    ``required_free_bytes`` for this card (``card`` carries the probe's
    ``MultiProcessorCount``, ``MaxThreadsPerMultiProcessor``, ``name`` and
    ``free_bytes``); the comparison is with the card's free memory now.

    ``levels`` is the case's vertical level count (:func:`mesh_levels`);
    ``None`` prices the leg on the port's default level count.

    A port tree that publishes no admission module, or one without the
    whole :data:`ADMISSION_SURFACE`, is not refused, since that would stop
    a run that may fit, and is not sized either: the receipt says
    ``sized: false`` and why.
    """

    import inspect

    row = PARENT_DEVICE_FOOTPRINT[parent_kind]
    root = Path(port_root).expanduser()
    admission, module_path = _port_admission(root, row)
    free = int(card["free_bytes"])
    if admission is None:
        return {"sized": False, "mesh": mesh, "cells": int(cells),
                "free_bytes": free,
                "reason": "the port tree publishes no device admission model; "
                          "looked in " + ", ".join(
                              str(root / relative)
                              for relative in row["admission_module"])}
    absent = [name for name in ADMISSION_SURFACE
              if not callable(getattr(admission, name, None))]
    if absent:
        return {"sized": False, "mesh": mesh, "cells": int(cells),
                "free_bytes": free, "model_module": str(module_path),
                "reason": f"the port's {module_path} has no "
                          f"{', '.join(absent)}, so it carries no per-card "
                          "capacity model this spine can price a leg with"}
    profile = admission.card_profile_from_attributes(str(card["name"]), card)
    by_levels = (levels is not None and "levels" in inspect.signature(
        admission.model_for_card).parameters)
    model = admission.model_for_card(
        profile, row["configuration"],
        **({"levels": int(levels)} if by_levels else {}))
    required = int(admission.required_free_bytes(int(cells), model))
    mib = 1024 ** 2
    receipt = {
        "sized": True, "mesh": mesh, "cells": int(cells),
        "card": str(card["name"]),
        "multiprocessors": int(profile.multiprocessors),
        "required_free_bytes": required, "free_bytes": free,
        "required_mib": round(required / mib, 1),
        "free_mib": round(free / mib, 1),
        "model_measured": bool(getattr(model, "measured", True)),
        "model_module": str(module_path),
        "configuration": row["configuration"],
        "levels": int(levels) if by_levels else None,
        "levels_basis": ("the case's level count" if by_levels
                         else "the port's default level count"),
    }
    if free < required:
        raise BridgeRefusal(
            f"the MPAS port's device stack for {mesh} ({int(cells):,} cells) "
            f"needs {required / mib:,.0f} MiB free on {card['name']} and "
            f"{free / mib:,.0f} MiB is free, so the leg would prepare the "
            "host state and then die inside the worker's first device "
            "allocation", **receipt,
            remedy="free the card, run on a card with more memory, or bind "
                   "a mesh with fewer cells")
    return receipt


def _segment_manifest_path(out_dir: Path, phase: str) -> Path:
    # ``seed`` writes two segments side by side; the caller asks for them
    # with read_segment().  Everything else writes one, in ``out``.
    if phase == "seed":
        return out_dir / "anchor" / SEGMENT_MANIFEST
    return out_dir / SEGMENT_MANIFEST


def read_segment(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a segment directory's manifest and check its member hashes."""
    from mpas_cycle_bridge import anchor_codec as codec

    root = Path(path).expanduser().absolute()
    manifest_path = root / SEGMENT_MANIFEST
    if not manifest_path.is_file():
        raise BridgeRefusal("no segment here", looked_for=str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in sorted((manifest.get("files") or {}).items()):
        member = root / name
        if not member.is_file():
            raise BridgeRefusal("segment member is missing",
                                segment=str(root), member=name)
        actual = codec.file_sha256(member)
        if actual != expected:
            raise BridgeRefusal(
                "segment member does not hash to what the worker recorded",
                segment=str(root), member=name,
                expected=expected, actual=actual)
    manifest["_dir"] = str(root)
    return manifest


def segment_arrays(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Load a segment's prognostics, sidecar, seam and vertical metric.

    The metric is REQUIRED and its absence is a refusal, not a default.
    A boundary without ``zz`` cannot have its exner recomputed by
    anything outside the port -- MPAS's equation of state is
    ``exner = (zz*(rd/p0)*rho_theta) ** (rd/cv)`` -- so a spine that
    quietly graded such a boundary on a unit metric would read a 0.407
    residual on a correct state, which is precisely what it did.
    """
    from mpas_cycle_bridge import anchor_codec as codec

    root = Path(manifest["_dir"])
    metric_path = manifest.get("metric_path")
    if not metric_path:
        raise BridgeRefusal(
            "segment carries no vertical metric; its exner cannot be "
            "recomputed and must not be graded on an assumed one",
            segment=str(root), have=sorted(manifest),
            remedy="re-run the leg with a worker that emits "
                   f"{codec.METRIC_STEM}; segments written before the "
                   "metric was carried cannot be re-graded")
    return {
        "prognostic": codec.read_arrays(root / manifest["prognostic_path"]),
        "derived": codec.read_arrays(root / manifest["derived_path"]),
        "seam": codec.read_arrays(root / manifest["seam_path"]),
        "metric": codec.read_arrays(root / metric_path),
    }


# --------------------------------------------------------------------------
# grading the evidence: what may this anchor be stamped?


#: The accurate label for a boundary whose state came out of the dycore but
#: whose analysis never went back INTO it.
FRAMES_KIND = "mpas-cuda-frames"
#: The label a boundary earns only when the analysis genuinely re-entered
#: the dycore and the device gave the analysed state back.
CLOSED_KIND = "mpas-cuda"


def stamp_for_segment(manifest: Mapping[str, Any], *,
                      steps_requested: int) -> dict[str, Any]:
    """Decide the parent kind from evidence, and say why.

    ``mpas-cuda`` is only earned when every one of these holds:

    * the worker executed the number of steps that were asked for --
      step receipts, counted, not a flag;
    * the device was rehydrated through ``_construct_device_stack``'s
      state/saved/backend_restart seam;
    * the hash of the state read back OFF THE DEVICE after rehydration
      equals the hash of the analysed state written to disk.  This is
      the one that cannot be faked by a run that crashed: if the device
      did not take the analysis, these differ, and no amount of
      downstream divergence rescues the claim;
    * woof in that process came from the pinned Arwen checkout.

    Anything short of all four is stamped ``mpas-cuda-frames``, with the
    gaps listed.  A stamp that overstates is worse than a missing
    feature: the next reader has no way to tell it apart from a real one.
    """
    gaps: list[str] = []
    executed = manifest.get("steps_executed")
    if not executed or int(executed) != int(steps_requested):
        gaps.append(
            f"step receipts: {executed!r} for {int(steps_requested)} steps "
            "requested; the dycore may not have run")
    if not manifest.get("rehydration", "").startswith("_construct_device_stack"):
        gaps.append("no rehydration through the port's state seam")
    rehydrated = manifest.get("rehydrated_sha256")
    analysed = manifest.get("analysed_sha256")
    if not rehydrated or not analysed:
        gaps.append("no post-rehydration device readback: there is no "
                    "evidence the analysis reached the dycore")
    elif rehydrated != analysed:
        gaps.append(
            "the state read back off the device after rehydration does not "
            f"match the analysed state on disk ({rehydrated[:16]} != "
            f"{analysed[:16]}): the analysis did NOT re-enter the dycore")
    if (manifest.get("arwen_pin") or {}).get("obeyed") is not True:
        gaps.append("woof did not come from the pinned WOOF-ArWen checkout")
    return {
        "parent_kind": CLOSED_KIND if not gaps else FRAMES_KIND,
        "evidence_gaps": gaps,
        "steps_executed": executed,
        "rehydrated_sha256": rehydrated,
        "analysed_sha256": analysed,
        "analysis_reentered_dycore": (bool(rehydrated) and
                                      rehydrated == analysed and not gaps),
    }


__all__ = ["ADMISSION_SURFACE", "BridgeRefusal", "CLOSED_KIND",
           "FRAMES_KIND", "PARENT_DEVICE_FOOTPRINT", "admit_parent_device",
           "bridge_root", "child_environment", "launch", "mesh_cells",
           "mesh_levels", "probe_card",
           "read_segment", "segment_arrays", "stamp_for_segment",
           "verify_bridge_purity", "worker_path"]
