"""Incremental native PNGs from verified compact stores, with bounded retrieval."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from woof import remote_artifacts as ra, remote_processed as legacy, remote_processed_v2 as viewer
from woof.remote_artifact_cache import Lease, _owned_directory

SCHEMA = "arwen.native-plots.v1"
STATUS_SCHEMA = "arwen.native-plot-progress.v1"
SELECTIONS_SCHEMA = "arwen.native-plot-selections.v1"
MAX_PANEL_BYTES = 32 * 1024**2
MAX_GALLERY_BYTES = 256 * 1024**2
#: The node renderer's own size bounds for one panel.
MIN_RENDER_PIXELS, MAX_RENDER_PIXELS = 256, 4096
RENDER_WIDTH, RENDER_HEIGHT = 1200, 900
#: How many distinct galleries one job keeps. A reader may hold a few sizes or
#: product sets at once; an unbounded set would render a job's whole history
#: once per spelling.
MAX_RENDER_SELECTIONS = 8
#: Render options this door carries. Every one of them keys the publication
#: identity, so two readers asking for two of them never overwrite each other.
RENDER_OPTIONS = ("profile", "products", "width", "height", "theme", "layout")
#: The job's record that the renderer drawing its galleries had no map
#: assets (:func:`_note_map_gap`), read into every status written after it.
MAP_GAP = "render-warning.json"
MAP_GAP_SCHEMA = "arwen.native-plot-render-warning.v1"


def _root(workspace, job):
    return _owned_directory(viewer._directory(viewer._root(workspace), job) / "native-plots")


_BUILTIN_THEMES = frozenset(("", "default", "light", "none", "classic", "dark", "woof-light", "woof-dark"))


def _theme_fingerprint(spec):
    """Bind a file theme's gallery to its JSON parents and declared assets.

    Rust validates and merges the theme. This reads only configuration
    metadata and file hashes, so an edit at the same path gets a new gallery.
    """
    if spec.strip().lower() in _BUILTIN_THEMES:
        return None
    files = []

    def visit(path, depth):
        path = Path(path).absolute()
        value, payload = ra._raw(path, viewer.MAX_METADATA_BYTES)
        if not isinstance(value, dict):
            raise ValueError("Native plot theme JSON must be an object")
        files.append({"path": str(path), "sha256": ra._sha(payload)})
        assets = {}
        parent = value.get("extends")
        if parent is not None:
            if not isinstance(parent, str):
                raise ValueError("Native plot theme extends must name a built-in or JSON file")
            if depth >= 8:
                raise ValueError("Native plot theme inheritance is deeper than eight themes")
            if parent.strip().lower() not in _BUILTIN_THEMES:
                assets.update(visit(path.parent / parent, depth + 1))
        for section, names in (("fonts", ("regular", "bold")), ("footer", ("logo",))):
            values = value.get(section)
            if isinstance(values, dict):
                for name in names:
                    text = values.get(name)
                    if isinstance(text, str):
                        assets[f"{section}.{name}"] = (path.parent / text).resolve()
        return assets

    assets = visit(Path(spec), 0)
    bound_assets = []
    for name, path in sorted(assets.items()):
        digest = ra._file_sha(path) if path.is_file() else None
        bound_assets.append({"field": name, "path": str(path), "sha256": digest})
    return ra._sha(ra._encoded({"files": files, "assets": bound_assets}))


def _check_theme_selection(selection):
    if ("theme_fingerprint" in selection
            and _theme_fingerprint(selection["theme"]) != selection["theme_fingerprint"]):
        raise ValueError("Native plot theme changed after gallery selection; request the gallery again to use the edited theme")


def render_selection(record, request=None):
    """One gallery selection, composed the same way at every door.

    A gallery is the product set this run selected, drawn at the size the
    request asked for. Both halves key the publication identity, so a reader
    at another size or another product set gets its own gallery instead of
    silently taking the one a different request published.
    """
    request = request or {}
    profile = request.get("profile", viewer.PROFILE)
    if "products" in request:
        selection = viewer._selection(profile, request["products"])
    elif profile == viewer.PROFILE:
        # The run's own set, read as the background preparer reads it, so a
        # run that asked only for sections gets no gallery of default maps.
        selection = viewer.job_selection(record)
    else:
        selection = viewer._selection(profile, viewer.map_selectors(record.get("products")))
    presentation = {}
    layout = request.get("layout")
    if layout not in (None, "auto", "fixed"):
        raise ValueError("Native plot layout must be auto or fixed")
    if layout == "auto":
        if request.get("width") is not None or request.get("height") is not None:
            raise ValueError("Native plot auto layout sizes the domain canvas; omit width and height, or use fixed layout")
        presentation["layout"] = "auto"
    theme = request.get("theme")
    if theme is not None:
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("Native plot theme must be a nonblank built-in name or a JSON file on the node")
        presentation["theme"] = theme.strip()
        fingerprint = _theme_fingerprint(presentation["theme"])
        if fingerprint is not None:
            presentation["theme_fingerprint"] = fingerprint
    size = {}
    for name, fallback in (("width", RENDER_WIDTH), ("height", RENDER_HEIGHT)):
        value = None if layout == "auto" else fallback if request.get(name) is None else request[name]
        if value is None:
            size[name] = None
            continue
        if type(value) is not int or not MIN_RENDER_PIXELS <= value <= MAX_RENDER_PIXELS:
            raise ValueError(f"Native plot {name} is {value!r} and the node's renderer draws panels "
                             f"from {MIN_RENDER_PIXELS} to {MAX_RENDER_PIXELS} pixels, so it would "
                             f"refuse the whole frame. Ask for a {name} inside that range.")
        size[name] = value
    identity = {"selection_id": selection["selection_id"], **size, **presentation}
    return {**selection, **size, **presentation,
            "render_id": ra._sha(ra._encoded(identity))}


def _presentation(selection):
    return {name: selection[name] for name in ("theme", "layout") if name in selection}



def _selection_root(root, selection):
    return _owned_directory(root / selection["render_id"])


def _receipt(root, sequence):
    return root / f"{ra._sequence(sequence):012d}.json"


def _registered(root, selection=None):
    """Every gallery selection this job serves, with its own first."""
    path = root / "selections.json"
    rows = []
    if path.exists():
        value, _ = ra._raw(path, viewer.MAX_METADATA_BYTES)
        if value.get("schema") == SELECTIONS_SCHEMA and isinstance(value.get("selections"), list):
            rows = [row for row in value["selections"] if isinstance(row, dict) and row.get("render_id")]
    if selection is not None and not any(row["render_id"] == selection["render_id"] for row in rows):
        rows = ([selection] + rows)[:MAX_RENDER_SELECTIONS]
        legacy._write(path, {"schema": SELECTIONS_SCHEMA, "selections": rows})
    return rows


def register(workspace, job, selection):
    """Record a gallery a reader asked for, so the watcher renders it too."""
    with Lease(_root(workspace, job) / "selections.lock", timeout=5) as lease:
        if lease.file is None:
            return _registered(_root(workspace, job))
        return _registered(_root(workspace, job), selection)


def status(workspace, job, selection=None):
    root = _root(workspace, job)
    path = (root if selection is None else _selection_root(root, selection)) / "status.json"
    return ra._raw(path, 64 * 1024)[0] if path.exists() else {
        "schema": STATUS_SCHEMA, "job_id": job, "state": "waiting_for_output", "ready": 0, "failed": 0}


def _published(root, job, event, authority):
    path = _receipt(root, event["sequence"])
    if not path.exists():
        return None
    value, _ = ra._raw(path, viewer.MAX_METADATA_BYTES)
    if (value.get("schema") != SCHEMA or value.get("job_id") != job
            or value.get("domain") != event["domain"] or value.get("sequence") != event["sequence"]
            or value.get("commit_sha256") != authority["sha256"]):
        raise ValueError("Native plots no longer match the committed forecast frame")
    return value


def _note_map_gap(plots_root):
    """Record, before a frame is drawn, that the renderer has no map assets.

    THE BREAKAGE: on a machine whose install lost its map files, every
    gallery picture comes back with no coastlines, borders or state lines
    and the job's status says nothing.  A run's own renders report this as
    a ``render_basemap_missing`` event; this watcher writes no events, so
    it keeps the same sentence in a record beside its status.  The record
    is kept once written, because the pictures already drawn still lack
    their maps.  Never raises: a check about the picture must not stop it.
    """
    path = plots_root / MAP_GAP
    try:
        if path.is_file():
            return
        from woof.render import BASEMAP_MISSING_CODE, renderer_basemap_gap
        gap = renderer_basemap_gap()
        if gap is not None:
            legacy._write(path, {"schema": MAP_GAP_SCHEMA, "code": BASEMAP_MISSING_CODE, "message": gap})
    except Exception:  # noqa: BLE001 - a status note never fails a gallery
        return


def _map_gap(plots_root):
    """The status field :func:`_note_map_gap` recorded, or nothing.

    ``render_warning`` is the field a run's own status fills from its
    ``render_basemap_missing`` event, so a reader shows both the same way.
    """
    path = plots_root / MAP_GAP
    try:
        if not path.is_file():
            return {}
        value, _ = ra._raw(path, 64 * 1024)
    except (OSError, ValueError):
        return {}
    message = value.get("message")
    if value.get("schema") != MAP_GAP_SCHEMA or not isinstance(message, str) or not message.strip():
        return {}
    return {"render_warning": " ".join(message.split())[:1600]}


def _render(root, record, bound, event, authority, entry, spacing, selection):
    _check_theme_selection(selection)
    from woof.render import require_renderer
    from woof.rustwx import renderer_env
    source = ra._inside(entry["source_path"], bound[0])
    if list(ra._stamp(source)) != entry["source_stamp"]:
        raise ValueError("WRF output changed after compact preparation")
    legacy._check_identity(entry["native_result"], bound[2]["run_id"], event, entry["source_sha256"])
    products = [row["slug"] for row in entry["products"] if row["available"]]
    output = root / f"frame-{event['sequence']:012d}"
    request_path, result_path = root / f"request-{event['sequence']:012d}.json", root / f"render-{event['sequence']:012d}.json"
    request = {"schema": "arwen.native-store-render-request.v1",
        "process_result": str(Path(entry["object_root"]) / "result.json"),
        "expected_frame_id": entry["frame"]["id"], "expected_source_sha256": entry["source_sha256"],
        "out_dir": str(output), "products": products, "spacing_m": spacing,
        "width": selection["width"], "height": selection["height"],
        **({"theme": selection["theme"]} if "theme" in selection else {})}
    legacy._write(request_path, request)
    with (root / f"render-{event['sequence']:012d}.log").open("ab", buffering=0) as log:
        # renderer_env hands an installed renderer the map files the
        # recast-woof-data package carries; without it a wheel's renderer finds
        # none and draws every gallery picture with no coastlines or borders.
        process = subprocess.run([str(require_renderer()), "--render-store-request", str(request_path),
            "--render-store-result", str(result_path)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            timeout=1800, check=False, env=renderer_env())
    if process.returncode:
        raise ValueError(f"Native plot rendering exited {process.returncode}; see {log.name}")
    result, _ = ra._raw(result_path, viewer.MAX_METADATA_BYTES)
    if (result.get("schema") != "arwen.native-store-render-result.v1"
            or result.get("frame_id") != entry["frame"]["id"]
            or result.get("identity") != entry["frame"]["identity"]
            or result.get("source_store_sha256") != entry["frame"]["rws_sha256"]
            or result.get("wrf_imported") is not False or result.get("volume_store_created") is not False):
        raise ValueError("Native renderer returned plots for another compact frame")
    panels = result.get("panels")
    if not isinstance(panels, list) or sorted(row.get("slug", "") for row in panels) != sorted(products):
        raise ValueError("Native renderer did not publish every available selected product")
    total = 0
    for panel in panels:
        path = ra._inside(panel["path"], output)
        if (path.suffix != ".png" or type(panel.get("bytes")) is not int
                or not 0 < panel["bytes"] <= MAX_PANEL_BYTES or path.stat().st_size != panel["bytes"]
                or ra._file_sha(path) != panel["sha256"]):
            raise ValueError("Native PNG checksum or size changed before publication")
        total += panel["bytes"]
    if total > MAX_GALLERY_BYTES:
        raise ValueError("Native frame gallery exceeds its transfer bound")
    if list(ra._stamp(source)) != entry["source_stamp"]:
        raise ValueError("WRF output changed during native rendering")
    _check_theme_selection(selection)
    _publish_receipt(output, panels, products)
    return {"schema": SCHEMA, "state": "ready", "job_id": record["id"], "run_id": bound[2]["run_id"],
        "domain": event["domain"], "sequence": event["sequence"], "valid_time": event["valid_time"],
        "commit_sha256": authority["sha256"], "frame_id": entry["frame"]["id"],
        "source_sha256": entry["source_sha256"], "source_stamp": entry["source_stamp"],
        "source_path": str(source), "panels": panels, "bytes": total,
        "render_id": selection["render_id"], "selection_id": selection["selection_id"],
        "selection_products": selection["products"], "width": result.get("width", selection["width"]),
        "height": result.get("height", selection["height"]), **_presentation(selection),
        "unavailable": [row for row in entry["products"] if not row["available"]],
        "processing": "existing_compact_store", "published_unix_ms": int(time.time() * 1000)}


def _publish_receipt(output, panels, products) -> None:
    """A render receipt beside the gallery, like every other delivery.

    The node-side gallery published pictures with no
    ``render-summary.json`` at all, so the desktop and remote surfaces
    that read that file saw nothing for a frame that rendered
    completely.  The panels are named by their own slug rather than by
    the layout's grammar, because this lane's filenames are the engine's
    store-request names and not ``woof render``'s.

    It is a RECORD.  A receipt that cannot be written is reported and
    never turns a finished render into a failure.
    """

    from woof import render_layout, render_receipts

    try:
        render_receipts.deliver(
            root=Path(output), engine="rust",
            requested_spec=",".join(products),
            written=[Path(panel["path"]) for panel in panels],
            failures=(), skipped=(), layout=render_layout.FLAT,
            families={str(Path(panel["path"]).resolve()): panel["slug"]
                      for panel in panels})
    except Exception as error:
        print(f"native plots: warning: no render receipt was published "
              f"({error})", file=sys.stderr)


def _spacing(record, domain):
    from woof.experiment import load_experiment
    viewer._initialization(record)  # verifies the saved config SHA before its spacing is read
    return load_experiment(record["snapshot_config"]).domain(domain).run.dx


def work_once(workspace, job, *, render=True, _completion=None, selection=None):
    from woof.remote_worker import TERMINAL
    record, state, bound, commits = legacy._job(
        workspace, job, **({"completion": True} if _completion is not None else {}))
    if _completion is not None:
        # Revalidate before any receipt, render or store request, including the
        # first pass that observes the wrapper's terminal result.
        _completion.validate(record, state, bound, commits)
    plots_root = _root(workspace, job)
    compact_root = viewer._root(workspace)
    own = render_selection(record)
    selection = own if selection is None else selection
    root = _selection_root(plots_root, selection)
    counts = {"committed": len(commits), "ready": 0, "failed": 0, "pending": 0, "panels": 0}
    if not viewer.has_map_products(selection):
        # No map to draw and no compact store to ask for: the note is the answer.
        summary = {"schema": STATUS_SCHEMA, "job_id": job, "simulation_state": state["state"], **counts,
                   "done": state["state"] in TERMINAL, "state": "no_map_products", "note": selection["note"],
                   "render_id": selection["render_id"], "selection_products": [],
                   "width": selection["width"], "height": selection["height"], **_presentation(selection),
                   "updated_unix_ms": int(time.time() * 1000)}
        if bound:
            summary["run_id"] = bound[2]["run_id"]
        legacy._write(root / "status.json", summary)
        if selection["render_id"] == own["render_id"]:
            legacy._write(plots_root / "status.json", summary)
        return summary
    candidate, wanted = None, None
    for event, authority in reversed(commits):
        published = _published(root, job, event, authority)
        if published is not None:
            if published["state"] not in ("ready", "failed"):
                raise ValueError("Invalid native plot publication state")
            counts[published["state"]] += 1
            counts["panels"] += len(published.get("panels", []))
            continue
        counts["pending"] += 1
        if candidate is None:
            entry = viewer._entry(compact_root, job, event, authority, selection)
            if viewer._available(entry):
                candidate = (event, authority, entry)
            elif entry is not None and entry["state"] in ("failed", "backpressure"):
                legacy._write(_receipt(root, event["sequence"]), {"schema": SCHEMA, "state": "failed", "job_id": job,
                    "domain": event["domain"], "sequence": event["sequence"], "commit_sha256": authority["sha256"],
                    "error": "Compact store preparation failed: " + entry.get("error", "unknown cause")})
            elif wanted is None and render and (entry is None or viewer._entry_state(entry) == "evicted"):
                # A long run may rotate the compact cache before plots catch up,
                # and a gallery for a product set the background preparer does
                # not hold has no store at all. Request only this one missing
                # compact frame, never a full store.
                wanted = {"domain": event["domain"], "sequence": event["sequence"],
                          "run_id": bound[2]["run_id"], "commit_sha256": authority["sha256"]}
    if wanted is not None and candidate is None:
        viewer.ensure(workspace, job, [{"profile": selection["profile"], "products": selection["products"],
                                        "selection_id": selection["selection_id"], **wanted}])
    done = state["state"] in TERMINAL and counts["pending"] == 0
    if candidate and render:
        # Before the status below is written, so the pass that draws the
        # first picture with no maps already says so.
        _note_map_gap(plots_root)
    summary = {"schema": STATUS_SCHEMA, "job_id": job, "simulation_state": state["state"], **counts,
        "done": done, "state": "complete_with_errors" if done and counts["failed"] else "complete" if done
            else "rendering" if candidate else "waiting_for_compact_stores",
        "render_id": selection["render_id"], "selection_products": selection["products"],
        "width": selection["width"], "height": selection["height"], **_presentation(selection),
        "updated_unix_ms": int(time.time() * 1000), **_map_gap(plots_root)}
    if bound:
        summary["run_id"] = bound[2]["run_id"]
    if candidate:
        event, authority, entry = candidate
        summary["active"] = {"domain": event["domain"], "sequence": event["sequence"], "valid_time": event["valid_time"]}
    legacy._write(root / "status.json", summary)
    if selection["render_id"] == own["render_id"]:
        # The job's own gallery is the one `woof remote status` reports.
        legacy._write(plots_root / "status.json", summary)
    if candidate and render:
        path = viewer._entry_path(compact_root, job, event["sequence"], selection)
        with Lease(viewer._directory(compact_root, job) / (path.stem + ".lock")) as lease:
            if lease.file is not None:
                if not viewer._available(entry):
                    return summary
                try:
                    spacing = _spacing(record, event["domain"])
                    value = _render(root, record, bound, event, authority, entry, spacing, selection)
                except Exception as error:
                    value = {"schema": SCHEMA, "state": "failed", "job_id": job, "domain": event["domain"],
                        "sequence": event["sequence"], "commit_sha256": authority["sha256"], "error": str(error)[:2000]}
                legacy._write(_receipt(root, event["sequence"]), value)
    return summary


def ensure(workspace, job):
    from woof import remote_worker as rw
    record = rw._record(rw._directory(workspace, job))
    if ra.plan_binding(record) is None:
        return
    root = _root(workspace, job)
    previous = root / "status.json"
    if previous.exists() and ra._raw(previous, 64 * 1024)[0].get("done"):
        return
    with Lease(root / "worker.lock") as lease:
        if lease.file is None:
            return
        environment = dict(os.environ)
        environment.pop(rw.TOKEN_ENV, None)
        environment.update(GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1", RAYON_NUM_THREADS="2",
            OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        with (root / "worker.log").open("ab", buffering=0) as log:
            subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--workspace", str(workspace),
                "--job", job], cwd=str(workspace), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True, close_fds=True)


def _receipt_state(root, job, state, error, *, done):
    legacy._write(root / "status.json", {"schema": STATUS_SCHEMA, "job_id": job, "state": state,
        "done": done, "error": str(error)[:2000]})


def worker(workspace, job, *, cancel=None):
    root = _root(workspace, job)
    with Lease(root / "worker.lock", timeout=3) as lease:
        if lease.file is None:
            return 0
        if hasattr(os, "nice"):
            os.nice(5)
        completion = ra.CompletionWait(workspace, job, cancel, time)
        while True:
            try:
                completion.check()
                completion.begin()
                value = None
                try:
                    value = _work_selections(workspace, job, completion)
                except ra.ProducerCompletionPending as pending:
                    # The runner exited and its wrapper has not settled. No
                    # receipt is written and no renderer runs in that window.
                    completion.pending(pending)
                except ra.ProducerCompletionUnprovable as unprovable:
                    # This job's runner-exit window cannot be proved and no
                    # renderer runs inside it either. A terminal receipt here
                    # would make ensure() refuse to relaunch this gallery for
                    # the life of the job, so the receipt stays non-terminal
                    # and says why the gallery is still pending.
                    completion.unprovable(unprovable)  # Raises if this wait held proof.
                    _receipt_state(root, job, "waiting_for_producer_completion",
                                   unprovable, done=False)
                if value is not None and value["done"]:
                    return 0
                completion.wait(.1 if value is not None and value.get("active") else 2)
            except ra.ProducerCompletionCancelled as cancelled:
                # ensure() refuses to relaunch on any done receipt, so a
                # cancelled gallery must stay non-terminal.
                _receipt_state(root, job, "cancelled", cancelled, done=False)
                return 2
            except Exception as error:
                _receipt_state(root, job, "failed", error, done=True)
                return 2


def _work_selections(workspace, job, completion=None):
    """One pass over every gallery this job serves; its own is always one."""
    from woof import remote_worker as rw
    record = rw._record(rw._directory(workspace, job))
    rows = _registered(_root(workspace, job), render_selection(record))
    summary = None
    for row in rows[:MAX_RENDER_SELECTIONS]:
        value = work_once(workspace, job, _completion=completion, selection=row)
        summary = value if summary is None or not value["done"] else summary
        summary["done"] = summary["done"] and value["done"]
    return summary


def catalog(request, workspace):
    allowed = {"schema", "action", "workspace", "job", "domain", "sequence", *RENDER_OPTIONS}
    if set(request) - allowed or request.get("action") != "native-plots":
        raise ValueError("Invalid native plot catalog request")
    domain = ra._domain(request.get("domain", 1))
    sequence = ra._sequence(request["sequence"]) if request.get("sequence") is not None else None
    job = request["job"]
    from woof import remote_worker as rw
    from woof.remote_preparation_v2 import ensure as prepare_stores
    selection = render_selection(rw._record(rw._directory(workspace, job)), request)
    root = _selection_root(_root(workspace, job), selection)
    register(workspace, job, selection)
    prepare_stores(workspace, job)
    ensure(workspace, job)
    value = {"schema": SCHEMA, "job_id": job, "domain": domain, "sequence": sequence,
        "waiting": True, "render_id": selection["render_id"], "selection_products": selection["products"],
        "selection_basis": (viewer.selection_basis(selection["profile"], selection["products"])
                            if viewer.has_map_products(selection) else selection["note"]),
        # False for a run that asked only for sections: its gallery never
        # publishes, so a reader answers with the basis note instead of
        # "still being prepared" for as long as it keeps asking.
        "map_products": viewer.has_map_products(selection),
        "width": selection["width"], "height": selection["height"], **_presentation(selection),
        "progress": status(workspace, job, selection)}
    try:
        record, _state, bound, commits = legacy._job_completing(workspace, job)
    except ra.ProducerCompletionPending:
        # The interactive door waits with the watcher instead of refusing for
        # the seconds a settling wrapper owns. No authority is published here.
        return {**value, "producer_completing": True}
    if bound is None:
        return value
    _, manifest_path, manifest, manifest_bytes, _started, binding = bound
    value.update(run_id=manifest["run_id"], run_manifest=ra._authority(manifest_path, manifest_bytes),
        remote_output_root=record["outdir"], run_root=str(ra.run_root(record)), remote_pid=manifest["pid"])
    if binding is not None:
        value["producer_binding"] = binding
    selected = [(event, authority) for event, authority in commits
        if event["domain"] == domain and (sequence is None or event["sequence"] == sequence)]
    if not selected:
        return value
    event, authority = selected[-1]
    value.update(sequence=event["sequence"], valid_time=event["valid_time"], commit=authority)
    published = _published(root, job, event, authority)
    if published is None:
        return value
    if published["state"] == "failed":
        value["error"] = published["error"]
        return value
    if published["run_id"] != manifest["run_id"] or list(ra._stamp(ra._inside(published["source_path"], bound[0]))) != published["source_stamp"]:
        raise ValueError("Native plot source changed after publication")
    value.update(waiting=False, frame_id=published["frame_id"], panels=published["panels"],
        unavailable=published["unavailable"], bytes=published["bytes"],
                 width=published["width"], height=published["height"])
    return ra._bounded(value)


def stream(request, workspace, output):
    fields = {"schema", "action", "workspace", "job", "domain", "sequence", "product",
        "expected_panel_sha256", "expected_commit_sha256", "expected_manifest_sha256"}
    if set(request) - (fields | set(RENDER_OPTIONS)) or fields - set(request) or request.get("action") != "stream-native-plot":
        raise ValueError("Invalid native plot stream request")
    if any(not ra.HEX.fullmatch(str(request[k])) for k in fields if k.startswith("expected_")):
        raise ValueError("Invalid native plot stream digest")
    query = {k: request[k] for k in ("schema", "workspace", "job", "domain", "sequence")}
    query.update({k: request[k] for k in RENDER_OPTIONS if k in request})
    value = catalog({**query, "action": "native-plots"}, workspace)
    if (value["waiting"] or value["commit"]["sha256"] != request["expected_commit_sha256"]
            or value["run_manifest"]["sha256"] != request["expected_manifest_sha256"]):
        raise ValueError("Native plot authority changed before transfer")
    panel = next((row for row in value["panels"] if row["slug"] == request["product"]), None)
    if panel is None or panel["sha256"] != request["expected_panel_sha256"] or not 0 < panel["bytes"] <= MAX_PANEL_BYTES:
        raise ValueError("Native plot product or checksum does not match the selected frame")
    path = ra._inside(panel["path"], _root(workspace, request["job"]))

    before = ra._stamp(path)
    digest, copied = hashlib.sha256(), 0
    with path.open("rb") as source:
        while block := source.read(min(1024**2, panel["bytes"] + 1 - copied)):
            copied += len(block)
            if copied > panel["bytes"]:
                raise ValueError("Native plot grew during transfer")
            digest.update(block)
            output.write(block)
    output.flush()
    if copied != panel["bytes"] or digest.hexdigest() != panel["sha256"] or ra._stamp(path) != before:
        raise ValueError("Native plot changed during transfer")


def _panel_name(slug):
    """A portable local file name for one panel of any selector family.

    A `var:`/`xsec:` selector is a legal product name and an illegal file name
    on one of the desktops this cache runs on, so the local name is derived
    once, here, rather than at each of the three places that touch the file.
    """
    return "".join(character if character.isalnum() or character in "._-" else "-"
                   for character in str(slug))[:160] + ".png"


def request_options(args):
    """The render options one desktop request carries, from the parsed door."""
    options = {}
    products = getattr(args, "products", None)
    if products is not None:
        options["products"] = viewer.selectors(products)
    for name in ("profile", "width", "height", "theme", "layout"):
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    return options


def sync(args, command, stream_command):
    from woof.remote_cli import _transport
    query = {"schema": "gpuwm.remote.request.v1", "action": "native-plots", "workspace": args.workspace,
        "job": args.job, "domain": ra._domain(args.domain), "sequence": ra._sequence(args.sequence),
        **request_options(args)}
    reply = _transport(command, query, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    value = reply.get("native_plots")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("job_id") != args.job or value.get("domain") != args.domain or value.get("sequence") != args.sequence:
        raise ValueError("Native plot gallery belongs to another selection")
    if value["waiting"]:
        return {"native_plots": value, "transferred_bytes": 0}
    panels = value.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("Invalid native plot gallery inventory: the node published a gallery with no panel list")
    if len(panels) > viewer.NODE_PRODUCT_LIMIT:
        # The bound is the node's own viewer profile, the same one that bounds
        # the selection this gallery was rendered from, so a selection the node
        # accepted is never refused on the way back.
        raise ValueError(f"The node's gallery inventory names {len(panels)} panels and its viewer profile "
                         f"renders at most {viewer.NODE_PRODUCT_LIMIT} named products, so this reply describes "
                         "no selection this door could have asked for; `woof remote list-products` prints "
                         "what the selected node's renderer serves.")
    names, total = set(), 0
    for panel in panels:
        if (not viewer.SELECTOR.fullmatch(str(panel.get("slug"))) or panel["slug"] in names
                or not ra.HEX.fullmatch(str(panel.get("sha256"))) or type(panel.get("bytes")) is not int
                or not 0 < panel["bytes"] <= MAX_PANEL_BYTES):
            raise ValueError("Invalid native plot gallery product, checksum or size")
        names.add(panel["slug"])
        total += panel["bytes"]
    if total > MAX_GALLERY_BYTES or total != value.get("bytes"):
        raise ValueError("Native plot gallery exceeds its transfer bound")
    # The gallery identity covers the render options too, so one cache root
    # holds one folder per selection rather than one per frame overwritten.
    key = ra._sha(ra._encoded([value["run_id"], value["commit"]["sha256"], value["render_id"], panels]))
    directory = _owned_directory(_owned_directory(Path(args.cache_root)) / key)
    transferred = 0
    for panel in panels:
        path = directory / _panel_name(panel["slug"])
        if path.exists():
            if path.is_symlink() or path.stat().st_size != panel["bytes"] or ra._file_sha(path) != panel["sha256"]:
                raise ValueError("A retained native plot changed; choose a fresh gallery cache")
        else:
            request = {**query, "action": "stream-native-plot", "product": panel["slug"],
                "expected_panel_sha256": panel["sha256"], "expected_commit_sha256": value["commit"]["sha256"],
                "expected_manifest_sha256": value["run_manifest"]["sha256"]}
            ra._download(stream_command, request, path, {"size_bytes": panel["bytes"], "sha256": panel["sha256"]})
            transferred += panel["bytes"]
    title = f"WOOF-ArWen native plots · d{args.domain:02d} · {value['valid_time']}"
    body = ''.join(f'<figure><a href="{html.escape(_panel_name(row["slug"]))}"><img loading="lazy" src="{html.escape(_panel_name(row["slug"]))}" alt="{html.escape(row["slug"])}"></a><figcaption>{html.escape(row["slug"].replace("_", " "))}</figcaption></figure>' for row in panels)
    gallery = directory / "index.html"
    gallery.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>' + html.escape(title)
        + '</title><style>body{font:16px system-ui;margin:24px;background:#eef2f6;color:#17252f}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}figure{margin:0;padding:12px;background:white}img{width:100%;height:auto}figcaption{padding:8px 0}h1{font-size:24px}</style><h1>'
        + html.escape(title) + '</h1><p>Native plots from this committed forecast frame. Click a plot to open its original PNG.</p><main>' + body + '</main>', encoding="utf-8")
    value.update(gallery_path=str(gallery), gallery_sha256=ra._file_sha(gallery))
    return {"native_plots": value, "transferred_bytes": transferred}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    from woof.remote_worker import _workspace
    return worker(_workspace({"workspace": args.workspace}), args.job,
                  cancel=ra.cancel_on_shutdown())


if __name__ == "__main__":
    raise SystemExit(main())
