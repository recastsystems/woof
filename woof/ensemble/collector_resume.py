"""Durable member diagnostic spools and exact precipitation endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from woof.ensemble.restart_roster import atomic_json, digest_file, relative_file, resolve_file

CONTRACT = "gpuwm-ensemble-collector-resume.v1"


def save(collector):
    from woof.io.classic_product import ClassicProduct
    root = collector.root
    directory = root / ".ensemble-resume"
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / "collector-arrays.nc.pending"
    arrays, rain, spools, files = {}, [], [], set()
    for number, (key, history) in enumerate(collector.rain_history.items()):
        entries = []
        for slot, (tick, value) in enumerate(history.items()):
            name = f"rain_{number}_{slot}"
            arrays[name] = value
            entries.append({"tick": tick, "variable": name})
        rain.append({"key": list(key), "initial_tick": collector.rain_initial_ticks[key], "entries": entries})
    ice = []
    for number, (key, history) in enumerate(getattr(getattr(collector, "ice", None), "history", {}).items()):
        row = {"key": list(key), "tick": history["tick"], "variables": {}}
        for field, value in history.items():
            if field == "tick":
                continue
            name = f"ice_{number}_{field}"
            arrays[name] = value
            row["variables"][field] = name
        ice.append(row)
    for number, (key, spool) in enumerate(collector.spools.items()):
        names = (f"latitude_{number}", f"longitude_{number}")
        arrays[names[0]], arrays[names[1]] = spool.latitude, spool.longitude
        spools.append({"key": list(key), "latitude": names[0], "longitude": names[1],
            "coordinate_file": spool.coordinate_file,
            "export_coordinates": spool._publish_coordinates})
        if spool.coordinate_file is not None:
            files.add(spool.coordinate_file["path"])
        files.add(relative_file(root, spool.manifest_path))
        for frame in spool.frames.values():
            files.update(frame["products"])
            files.update(frame["maps"])
            if frame["status"] == "pending":
                files.update(relative_file(root, pack["path"]) for pack in frame["packs"])
    if arrays:
        with ClassicProduct(pending) as writer:
            writer.setncattr("collector_resume_contract", CONTRACT)
            for number, (name, array) in enumerate(arrays.items()):
                array = np.asarray(array)
                dims = (f"y_{number}", f"x_{number}")
                writer.createDimension(dims[0], array.shape[0])
                writer.createDimension(dims[1], array.shape[1])
                writer.createVariable(name, str(array.dtype), dims)[:] = array
        path = directory / (digest_file(pending) + ".nc")
        pending.replace(path)
        files.add(relative_file(root, path))
        array_file = relative_file(root, path)
    else:
        array_file = None
    archive = None if collector.member_archive is None else collector.member_archive.receipt()
    if archive is not None:
        files.add(relative_file(root, collector.member_archive.path))
        files.update(row["path"] for row in archive["files"])
    ledger = collector.history_ledger.validate_retirements()
    retired = {row["path"] for row in ledger if row["retirement_state"] == "retired"}
    files.update(row["path"] for row in collector.member_files if row["path"] not in retired)
    if ledger:
        files.add(relative_file(root, collector.history_ledger.path))
        files.update(row["retirement_receipt"] for row in ledger if "retirement_receipt" in row)
    document = {"schema": CONTRACT, "member_order": list(collector.member_order),
        "member_metadata": list(collector.member_metadata), "start_time": collector.start_time.isoformat(),
        "requests": [request.describe() for request in collector.requests],
        "arrays": array_file, "rain": rain, "spools": spools,
        "rain_relocations": collector.rain_relocations,
        "rain_placements": [[list(key), list(value)] for key, value in getattr(collector, "_placements", {}).items()],
        "ice": ice, "ice_intervals_hours": sorted(getattr(getattr(collector, "ice", None), "intervals", set())) ,
        "output_ticks": [[list(key), tick] for key, tick in collector.output_ticks],
        "member_files": collector.member_files, "archive": archive, "files": sorted(files),
        "replay": {"gpu_replay": collector.gpu_replay, "defer_replay": collector.defer_replay,
            "render_products": list(collector.render_products)},
        "history_ledger": ledger}
    atomic_json(directory / "collector.json", document)
    retire_superseded_arrays(root, keep=array_file)
    return document


def retire_superseded_arrays(root, *, keep):
    """Delete the arrays snapshots an earlier save wrote and nothing names any more.

    THE BREAKAGE THIS PREVENTS: every save wrote a new digest-named arrays file
    and never retired the previous one, about 100 MB per checkpoint per domain.
    The 30-member, 48 h ice-storm ensemble (test host, 2026-10-21) held 1,039 of them,
    59.9 GB, while the live collector.json named one.  The current file, and
    any file the sealed restart roster names, are kept.  Each roster generation
    snapshot holds its own link, so a sealed roster still restores.
    """
    import re
    directory = Path(root) / ".ensemble-resume"
    named = set()
    roster = Path(root) / "ensemble-restart.json"
    if roster.is_file():
        named.update(re.findall(r"[0-9a-f]{64}\.nc", roster.read_text(encoding="utf-8")))
    if keep is not None:
        named.add(Path(keep).name)
    removed = []
    for path in directory.glob("*.nc"):
        if re.fullmatch(r"[0-9a-f]{64}\.nc", path.name) and path.name not in named:
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def restore(collector):
    from woof.netcdf_bridge import Dataset
    from woof.ensemble.batch_product_output import NativeDiagnosticSpool
    root = collector.root
    document = json.loads((root / ".ensemble-resume" / "collector.json").read_text(encoding="utf-8"))
    expected = {"schema": CONTRACT, "member_order": list(collector.member_order),
        "member_metadata": list(collector.member_metadata), "start_time": collector.start_time.isoformat(),
        "requests": [request.describe() for request in collector.requests],
        "replay": {"gpu_replay": collector.gpu_replay, "defer_replay": collector.defer_replay,
            "render_products": list(collector.render_products)}}
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("ensemble output continuation has different members, seeds, products or accumulation start")
    arrays = {}
    if document["arrays"] is not None:
        with Dataset(resolve_file(root, document["arrays"])) as source:
            source.set_auto_maskandscale(False)
            arrays = {name: np.asarray(variable[:], dtype=variable.dtype) for name, variable in source.variables.items()}
    for row in document["rain"]:
        key = tuple(row["key"])
        collector.rain_initial_ticks[key] = row["initial_tick"]
        collector.rain_history[key] = {entry["tick"]: arrays[entry["variable"]] for entry in row["entries"]}
    collector.rain_relocations = document.get("rain_relocations", [])
    collector._placements = {tuple(key): tuple(value) for key, value in document.get("rain_placements", [])}
    # Older snapshots did not store placement state. Placement sequence is part
    # of a moving nest's counter identity: recover the latest visit, including
    # A -> B -> A, ordered by sequence then clock (relocation may capture both
    # footprints at the same model clock), instead of comparing a return to an
    # old footprint. Four-part keys from older checkpoints count as sequence 0.
    placements = {}
    for key, history in collector.rain_history.items():
        tick = max((entry for entry in history if isinstance(entry, int)), default=-1)
        sequence = key[4] if len(key) > 4 else 0
        order = (sequence, tick)
        if key[:3] not in placements or order > placements[key[:3]][0]:
            placements[key[:3]] = (order, key[3])
    for key, (order, geometry) in placements.items():
        collector._placements.setdefault(key, (geometry, order[0]))
    for row in document.get("ice", []):
        collector.ice.history[tuple(row["key"])] = {"tick": row["tick"],
            **{field: arrays[name] for field, name in row["variables"].items()}}
    if hasattr(collector, "ice"):
        collector.ice.intervals.update(document.get("ice_intervals_hours", []))
    collector.output_ticks = {(tuple(key), tick) for key, tick in document["output_ticks"]}
    collector._resume_output_ticks = set(collector.output_ticks)
    collector._resume_rain_ticks = {key: max((tick for tick in history if isinstance(tick, int)), default=-1)
        for key, history in collector.rain_history.items()}
    collector.member_files = document["member_files"]
    if collector.history_ledger.inventory() != document.get("history_ledger", []):
        raise ValueError("retained history ledger differs from the collector checkpoint authority")
    completed, pending = [], []
    for row in document["spools"]:
        key = tuple(row["key"])
        spool = NativeDiagnosticSpool(root, members=collector.members, requests=collector.requests,
            latitude=arrays[row["latitude"]], longitude=arrays[row["longitude"]], renderer=collector.renderer,
            domain=key[0], keep_member_files=collector.keep_member_files, tile_rows=collector.tile_rows,
            events={name: [{"field": c.field, "units": c.units, "threshold": c.threshold,
                "comparison": c.comparison} for c in conditions] for name, conditions in collector.events.items()},
            member_order=collector.member_order, member_metadata=collector.member_metadata,
            export_coordinates=row["export_coordinates"], gpu_replay=collector.gpu_replay, resume=True)
        coordinate = row["coordinate_file"]
        if coordinate is not None:
            from woof.ensemble.product_worker import _identity, _owned_path
            _identity(_owned_path(root, coordinate["path"]), expected=coordinate)
        spool.coordinate_file = coordinate
        collector.spools[key] = spool
        for frame in spool.frames.values():
            domain = spool.domain.split("-grid-", 1)[0]
            grid, _, episode = domain.removeprefix("d").partition("-episode-")
            cohort = collector.cohorts.setdefault((int(grid), int(episode or 0), frame["valid_time"]), {})
            cohort[key[1]] = {"members": set(frame["members_received"]), "spool": spool}
            if frame["status"] == "complete":
                completed.append((spool.domain, frame["valid_time"]))
            elif frame["status"] == "pending" and frame["members_received"] == sorted(collector.member_order):
                pending.append((spool, frame["valid_time"]))
    collector.product_consumer.restore_completed(completed)
    if not collector.defer_replay:
        for spool, valid in pending:
            collector._schedule_product(spool, valid)
    return document
