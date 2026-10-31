"""Table-driven history sidecars shared by both public engines.

Only filenames, configuration and receipts cross this Python boundary.
Rust reads, chunks, compresses and commits the native arrays. A worker
observes publication ledgers, never file age or a partly written frame.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib

TABLE = Path(__file__).with_name("data") / "output_formats.json"
ABI_MARKER = ("rw_icechunk --request REQUEST.json schema=icechunk-output.request/v1 "
              "modes=run,append,finalize progress=jsonl native-arrays=v1")


def table():
    return json.loads(TABLE.read_text(encoding="utf-8"))


def selected(value=None):
    document = table()
    values = document["default"] if value is None else value
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("output formats must be a nonempty list of table row names")
    names = [str(v).strip() for v in values]
    rows = {r["id"]: r for r in document["rows"]}
    unknown = set(names) - rows.keys()
    if unknown:
        raise ValueError(f"unknown output formats {sorted(unknown)}; choose {sorted(rows)}")
    # Sidecars consume ordinary history. Selecting a sidecar retains that source.
    return [rows[n] for n in dict.fromkeys(names)]


def find_binary(row=None):
    from woof import bridges
    row = row or next(r for r in table()["rows"] if r["writer"])
    name = bridges.executable_name(row["writer"])
    if not os.environ.get(row["env"]):
        root = Path(__file__).resolve().parent.parent / row["crate"] / "target"
        for kind in ("release", "debug"):
            path = root / kind / name
            if path.is_file():
                return bridges.accept_resolved(path.resolve())
    found = bridges.find_artifact(row["env"], name, crate_relative=row["crate"])
    # On demand since 2.9.10 (PyPI per-file limit): a run that asks for the store stages the
    # writer from this release's bundle on first use. Never from doctor (inspection only).
    if found is None and not os.environ.get(row["env"]) and bridges.fetch_on_demand(row["writer"]):
        found = bridges.find_artifact(row["env"], name, crate_relative=row["crate"])
    return found


def remedy(row=None):
    """How to get the writer, as woof doctor prints remedies: every line a command or a # comment.

    One implementation with every other bundled artifact (bridges.artifact_remedy): the checkout's offline
    in-crate build, the release bundle's one-file fetch, or the clone-and-build route on a pip install.
    Breakage it prevents: the prose remedy ("Install a platform wheel ..., or build cargo build ...
    --manifest-path tools/rw_icechunk/Cargo.toml ...") broke doctor's paste contract (tests/test_doctor.py,
    42 reds at the 2.9.10 EB4 tip 00c11b5330), and a --manifest-path build from the repository root
    bypasses the vendored registry tools/rw_icechunk/.cargo/config.toml names.
    """
    row = row or next(r for r in table()["rows"] if r["writer"])
    from woof import bridges
    why = bridges.on_demand_failure(row["writer"])
    return ((f"# {row['writer']} was not staged on demand: {why}\n" if why else "")
            + bridges.artifact_remedy(
                env_var=row["env"], filename=bridges.executable_name(row["writer"]),
                subject=f"the Icechunk history writer ({row['writer']})",
                crate_relative=row["crate"], artifact=row["writer"]))


def require_binary(row):
    from woof import bridges
    path = find_binary(row)
    if path is None:
        raise ValueError(f"{row['id']} output needs {row['writer']}; the run would finish "
                         f"without its requested store. {remedy(row)}")
    ok, detail = bridges.bridge_abi_matches(row["writer"], path)
    if not ok:
        raise ValueError(f"{row['writer']} cannot read this output request: {detail}. {remedy(row)}")
    return path


def capabilities():
    rows = []
    for row in table()["rows"]:
        item = dict(row, available=True)
        if row["writer"]:
            try:
                require_binary(row)
            except (ValueError, FileNotFoundError) as error:
                item.update(available=False, reason=str(error))
        rows.append(item)
    return {"schema": "gpuwm.output-formats/v1", "rows": rows}


def add_arguments(parser):
    parser.add_argument("--output-config", type=Path, default=None, metavar="TOML",
                        help="Read [output] formats and icechunk_store from TOML; CLI values override it")
    parser.add_argument("--output-format", default=None, metavar="FORMATS",
                        help="Output-format table rows, comma separated (default netcdf); "
                             "netcdf,icechunk adds atomic native Zarr v3 history")
    parser.add_argument("--icechunk-store", default=None, metavar="URL",
                        help="Output store base: local path or s3://bucket/prefix; "
                             "domain and member paths are appended (default RUN/icechunk)")


def options(arguments):
    config = (getattr(arguments, "output_config", None) or getattr(arguments, "config", None) or getattr(arguments, "experiment_config", None))
    block = {}
    if config and Path(config).is_file() and Path(config).suffix == ".toml":
        with Path(config).open("rb") as stream:
            block = tomllib.load(stream).get("output", {})
    formats = getattr(arguments, "output_format", None)
    if formats is None:
        formats = block.get("formats")
    store = getattr(arguments, "icechunk_store", None)
    if store is None:
        store = block.get("icechunk_store")
    rows = selected(formats)
    if store is not None and not any(r["writer"] for r in rows):
        raise ValueError("--icechunk-store needs an output format with a store writer")
    return rows, store


def writer_launch(row, command):
    """Give background writers a bounded CPU share without forking Python hooks.

    This is called from the publication watcher thread, so preexec_fn is unsafe.
    The priority launchers exec the writer and its threads inherit their policy.
    """
    policy = row.get("scheduling", {})
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if hasattr(os, "sched_getaffinity"):
        cpu_count = min(cpu_count, len(os.sched_getaffinity(0)))
    limit = max(1, int(cpu_count * policy.get("cpu_fraction", 0.5)))
    environment = os.environ.copy()
    for variable in ("RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
        # An inherited smaller pool is respected; a larger one cannot undo the cap.
        requested = int(environment.get(variable, limit))
        environment[variable] = str(min(limit, requested) if requested > 0 else limit)
    kwargs = {"env": environment}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        nice = shutil.which("nice")
        if nice is None:
            raise RuntimeError("background history output needs the nice utility to "
                               "keep its CPU priority below the model")
        command = [nice, "-n", str(policy.get("nice", 10)), *command]
        ionice = shutil.which("ionice")
        if ionice is not None:
            command = [ionice, "-c", "2", "-n", str(policy.get("io_priority", 7)), *command]
    return command, kwargs


class HistoryOutputs:
    """A path-only polling worker; close drains it after the model finishes.

    It reads the writer's publication journal or durable sidecar. A netCDF
    path existing by itself is insufficient. Failed exports are surfaced at
    close and preserved in the receipt, never converted to successful runs.
    """
    def __init__(self, root, rows, store=None, *, engine="woof"):
        self.root = Path(root).resolve()
        self.rows = [(r, require_binary(r)) for r in rows if r["writer"]]
        self.store = store
        self.engine = engine
        self.seen = set()
        self.repositories = set()
        self.events = []
        self.error = None
        self.success = True
        self.stop = threading.Event()
        self.thread = None

    def finish(self, result):
        self.success = result in (None, 0)
        return result

    def __enter__(self):
        if self.rows:
            self.thread = threading.Thread(target=self._work, name="history-output-formats", daemon=True)
            self.thread.start()
        return self

    def _published(self):
        paths = set()
        if not self.root.is_dir():
            return []
        journals, markers = [], []
        for directory, folders, files in os.walk(self.root):
            folders[:] = [n for n in folders if n not in ("icechunk", "output-formats", "scratch", "cache", ".git", ".frames")]
            if "icechunk" in files or "zarr.json" in files:
                folders[:] = []
            folder = Path(directory)
            if ".history-written" in files:
                journals.append(folder / ".history-written")
            markers.extend(folder / n for n in files if n.endswith(".durable.json"))
        for ledger in journals:
            for name in ledger.read_text(encoding="utf-8").splitlines():
                if name.startswith("wrfout_d") and Path(name).name == name:
                    paths.add(ledger.parent / name)
        for marker in markers:
            # Hex history, not diagnostics or restart streams.
            name = marker.name.removesuffix(".durable.json")
            if not name.startswith(("history.", "output.", "cuda-history.")):
                if self.engine != "woof hex":
                    continue
                from woof.hex.forecast_door import history_files
                if marker.with_name(name) not in history_files(marker.parent):
                    continue
            record = json.loads(marker.read_text(encoding="utf-8"))
            path = marker.with_name(name)
            info = path.stat()
            if info.st_size != record["bytes"] or info.st_mtime_ns != record["mtime_ns"]:
                raise RuntimeError(f"published history changed: {path}")
            paths.add(path)
        return sorted(paths, key=lambda p: (p.name, str(p)))

    def _store_for(self, row, path):
        match = re.search(r"wrfout_(d\d+)_", path.name)
        domain = match.group(1) if match else "mesh"
        if self.store is None:
            return str(path.parent / "icechunk" / domain)
        # Keep independent member, segment and domain histories independent.
        relative = path.parent.relative_to(self.root)
        parts = [p for p in relative.parts if p not in ("run", "history", "wrfout", "chain")]
        return "/".join([str(self.store).rstrip("/"), *parts, domain])

    def _invoke(self, row, binary, store, inputs, mode):
        receipt_dir = self.root / "output-formats"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        request = {"schema": row["request_schema"], "mode": mode, "store": store,
                   "inputs": [str(p) for p in inputs], "chunks": row["chunks"]}
        with tempfile.TemporaryDirectory(prefix="request-", dir=receipt_dir) as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            started = time.monotonic()
            command, launch = writer_launch(row, [str(binary), "--request", str(path)])
            result = subprocess.run(command, capture_output=True, text=True, **launch)
            for line in result.stdout.splitlines():
                event = json.loads(line)
                event["format"] = row["id"]
                self.events.append(event)
                with (receipt_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
            if result.returncode:
                raise RuntimeError(f"{row['writer']} failed: {result.stderr.strip()}")
            self.events.append({"event": "stage", "format": row["id"],
                                "wall_seconds": time.monotonic()-started, "mode": mode})

    def _scan(self):
        for path in self._published():
            if path in self.seen:
                continue
            for i, (row, binary) in enumerate(self.rows):
                store = self._store_for(row, path)
                self._invoke(row, binary, store, [path], "append")
                self.repositories.add((i, store))
            self.seen.add(path)

    def _work(self):
        try:
            while not self.stop.is_set():
                self._scan()
                self.stop.wait(0.2)
            self._scan()
        except BaseException as error:
            self.error = error

    def __exit__(self, kind, error, traceback):
        if self.thread is None:
            return False
        self.stop.set()
        self.thread.join()
        try:
            if self.error is not None:
                raise self.error
            if kind is None and self.success:
                if not self.seen:
                    raise RuntimeError("requested output formats received no published history frames; no store was produced")
                for i, store in sorted(self.repositories):
                    row, binary = self.rows[i]
                    self._invoke(row, binary, store, [], "finalize")
        except BaseException as failure:
            self.error = failure
            raise
        finally:
            if self.root.is_dir():
                folder = self.root / "output-formats"
                folder.mkdir(exist_ok=True)
                (folder / "receipt.json").write_text(json.dumps({
                    "schema": "gpuwm.output-formats.receipt/v1", "engine": self.engine,
                    "status": "failed" if self.error or kind or not self.success else "complete",
                    "error": str(self.error) if self.error else None,
                    "frames": len(self.seen), "events": self.events}, indent=2), encoding="utf-8")
        elapsed = sum(e.get("wall_seconds", 0) for e in self.events if e["event"] == "stage")
        print(f"[output-formats] {len(self.seen)} history files, {elapsed:.3f} s writer wall; "
              f"receipt {self.root / 'output-formats' / 'receipt.json'}", flush=True)
        return False


def argument_values(arguments, *, relative=None):
    """Forward the public options through an orchestrating door."""
    result = []
    for name in ("output_format", "icechunk_store", "output_config"):
        value = getattr(arguments, name, None)
        if value is not None:
            result.extend(("--" + name.replace("_", "-"), str(value)))
    if relative is not None:
        _, store = options(arguments)
        if store is not None:
            if "--icechunk-store" in result:
                at = result.index("--icechunk-store")
                del result[at:at+2]
            result.extend(("--icechunk-store", str(store).rstrip("/") + "/" + str(relative)))
    return result


@contextmanager
def command_scope(arguments, *, engine="woof"):
    if engine == "woof" and getattr(arguments, "command", None) not in (
            "run", "go", "ensemble", "downscale", "resume", "branch", "sim"):
        yield
        return
    rows, store = options(arguments)
    if any(getattr(arguments, name, False) for name in ("dry_run", "prepare_only", "print_command", "readiness")):
        for row in rows:
            if row["writer"]:
                require_binary(row)
        yield None
        return
    if not any(r["writer"] for r in rows):
        yield
        return
    restart = getattr(arguments, "restart", None)
    if store is None and engine == "woof" and restart:
        previous = Path(restart).resolve().parent / "wrfout" / "icechunk"
        if previous.is_dir():
            store = str(previous)
    # Ensemble aggregation normally retires or omits raw frames. A native
    # store explicitly asks for those frames, so retain their source history.
    if engine == "woof":
        ensemble = getattr(arguments, "members", None)
        config = getattr(arguments, "config", None) or getattr(arguments, "experiment_config", None)
        if config and Path(config).is_file() and Path(config).suffix == ".toml":
            with Path(config).open("rb") as stream:
                ensemble = ensemble or tomllib.load(stream).get("ensemble")
        if ensemble:
            arguments.keep_member_files = True
    root = getattr(arguments, "outdir", None) or getattr(arguments, "out", None)
    if root is None:
        config = (getattr(arguments, "output_config", None) or getattr(arguments, "config", None) or getattr(arguments, "experiment_config", None))
        if config:
            root = Path(config).with_name(Path(config).stem + "-go")
        else:
            raise ValueError("output formats need a run output directory")
    root = Path(root).resolve()
    inherited = os.environ.get("WOOF_OUTPUT_SCOPE_ROOT")
    if inherited and root.is_relative_to(Path(inherited)):
        yield None
        return
    with HistoryOutputs(root, rows, store, engine=engine) as scope:
        os.environ["WOOF_OUTPUT_SCOPE_ROOT"] = str(root)
        try:
            yield scope
        finally:
            if inherited is None:
                os.environ.pop("WOOF_OUTPUT_SCOPE_ROOT", None)
            else:
                os.environ["WOOF_OUTPUT_SCOPE_ROOT"] = inherited
