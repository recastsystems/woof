"""Small WOOF-ArWen CLI client. Queries and inspection only; no job mutations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


class InterfaceError(RuntimeError):
    pass


def query(python: str, arguments: list[str], schema: str, *, cwd: Path | None = None) -> dict:
    result = subprocess.run(
        [python, "-m", "woof.cli", *arguments], cwd=cwd,
        capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    if result.returncode:
        raise InterfaceError(result.stderr.strip() or result.stdout.strip() or f"CLI exit {result.returncode}")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise InterfaceError("Query did not return one JSON document; retain stderr for diagnosis") from error
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise InterfaceError(f"Expected {schema}; update the client for the returned contract")
    if document.get("ok") is False or document.get("error"):
        raise InterfaceError(str(document.get("error") or document.get("message") or "Query refused"))
    return document


class RunReader:
    """Read a pinned producer's heartbeat and only complete new event records."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        payload = self.manifest_path.read_bytes()
        self.manifest_digest = hashlib.sha256(payload).hexdigest()
        self.manifest = json.loads(payload)
        if not isinstance(self.manifest, dict) or self.manifest.get("schema") != "gpuwm.run-manifest.v1":
            raise InterfaceError("Unsupported run manifest")
        self.run_id = self.manifest.get("run_id")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise InterfaceError("Run manifest has no run identity")
        self.events_path = self._path("events_path", "events.jsonl")
        self.progress_path = self._path("progress_path", "run-progress.json")
        self.offset = 0
        self.last_sequence = -1

    def _path(self, key: str, fallback: str) -> Path:
        value = self.manifest.get(key)
        if value is None:
            return self.manifest_path.parent / fallback
        path = Path(value)
        if not path.is_absolute():
            raise InterfaceError(f"Manifest {key} must be absolute")
        return path

    def snapshot(self) -> dict:
        digest = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        if digest != self.manifest_digest:
            raise InterfaceError("The producer manifest changed; reconnect to the intended run")
        progress = None
        if self.progress_path.exists():
            progress = json.loads(self.progress_path.read_bytes())
            if not isinstance(progress, dict) or progress.get("schema") != "gpuwm.run-progress/v1":
                raise InterfaceError("Unsupported run progress document")
            if progress.get("run_id") != self.run_id:
                raise InterfaceError("Progress belongs to another run")
        events = []
        if self.events_path.exists():
            with self.events_path.open("rb") as stream:
                if stream.seek(0, 2) < self.offset:
                    raise InterfaceError("The durable event stream was truncated")
                stream.seek(self.offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        self.offset = start
                        break
                    if not line.strip():
                        self.offset = stream.tell()
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise InterfaceError("Durable event record must be a JSON object")
                    sequence = event.get("sequence")
                    if (event.get("schema_version") != "gpuwm.run-plan.event.v1"
                            or type(sequence) is not int or sequence <= self.last_sequence):
                        raise InterfaceError("Unexpected event schema or nonmonotonic sequence")
                    if (not isinstance(event.get("event"), str) or not event["event"]
                            or type(event.get("emitted_unix_ms")) is not int or event["emitted_unix_ms"] < 0):
                        raise InterfaceError("Durable event needs its event tag and emission timestamp")
                    if event.get("run_id", self.run_id) != self.run_id:
                        raise InterfaceError("Event belongs to another run")
                    self.last_sequence = sequence
                    self.offset = stream.tell()
                    events.append(event)
        return {"run_id": self.run_id, "manifest_sha256": self.manifest_digest,
                "progress": progress, "new_events": events, "next_byte": self.offset}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Interpreter from the WOOF-ArWen installation")
    parser.add_argument("--cwd", type=Path, help="Optional runtime working directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources")
    commands.add_parser("products")
    commands.add_parser("physics")
    commands.add_parser("inventory")
    review = commands.add_parser("review")
    review.add_argument("plan", type=Path)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            output = RunReader(args.manifest).snapshot()
        elif args.command == "review":
            plan = str(args.plan.resolve())
            output = {mode: query(args.python, ["run-plan", plan, f"--{mode}"],
                                 f"gpuwm.run-plan.{'resolved' if mode == 'resolve' else 'estimate'}.v1", cwd=args.cwd)
                      for mode in ("resolve", "estimate")}
        else:
            arguments, schema = {
                "sources": (["sources", "--json"], "gpuwm.run-plan.sources.v1"),
                "products": (["run-plan", "--catalog"], "gpuwm.run-plan.catalog.v1"),
                "physics": (["run-plan", "--physics-profiles"], "gpuwm.run-plan.physics-profiles.v1"),
                "inventory": (["run-plan", "--probe", "--no-readiness"], "gpuwm.run-plan.probe.v1"),
            }[args.command]
            output = query(args.python, arguments, schema, cwd=args.cwd)
        print(json.dumps(output, indent=2))
        return 0
    except (InterfaceError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
