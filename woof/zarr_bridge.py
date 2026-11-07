"""Orchestration for the native regular-grid Zarr acquisition bridge."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from woof.bridges import (BRIDGE_ABI_MARKERS, DecoderContractError, accept_resolved,
                           artifact_remedy, bridge_abi_matches, default_bridge_dir,
                           legacy_bridge_candidates,
                           executable_name, packaged_bridge_dir)

CRATE_RELATIVE = "tools/zarr_bridge"
BRIDGE_ENV = "WOOF_RW_ZARR"
ABI_MARKER = BRIDGE_ABI_MARKERS["rw_zarr"]


def _remedy(filename: str) -> str:
    return artifact_remedy(env_var=BRIDGE_ENV, filename=filename,
                           subject="the native Zarr reader", crate_relative=CRATE_RELATIVE,
                           artifact="rw_zarr")


def resolve_zarr_bin() -> Path:
    """The Zarr reader this tree's fetch launches, or a refusal saying why not.

    A reader that exists but predates this tree's reader contract is
    passed over for the next rung, and refused when no rung holds a
    current one.  Existence alone is not the question: a reader staged
    by an older release launched, then refused every ARCO ERA5 request
    with 'level: source units "Hectopascal(hPa)" disagree with declared
    units "hPa"', a spelling the current reader reads through its units
    table.
    """
    filename = executable_name("rw_zarr")
    override = os.environ.get(BRIDGE_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"WOOF_RW_ZARR names a missing file: {path}")
        candidates = (path,)
    else:
        root = Path(__file__).resolve().parent.parent
        candidates = (root / CRATE_RELATIVE / "target/release" / filename,
                      root / CRATE_RELATIVE / "target/debug" / filename,
                      root / "libexec/bridges" / filename,
                      packaged_bridge_dir() / filename,
                      default_bridge_dir() / filename,
                      *legacy_bridge_candidates(filename))
    stale = []
    for path in candidates:
        if not path.is_file():
            continue
        current, evidence = bridge_abi_matches("rw_zarr", path)
        if current:
            return accept_resolved(path.resolve())
        stale.append(f"{path} {evidence}")
    if not stale and not override:
        # On demand since WOOF 1.1.6 (PyPI per-file limit): pinned in this release's
        # bundle and staged by the first use (never inside bridges.inspection_only()).
        from woof.bridges import fetch_on_demand
        staged = default_bridge_dir() / filename
        if fetch_on_demand("rw_zarr") and staged.is_file():
            current, evidence = bridge_abi_matches("rw_zarr", staged)
            if current:
                return accept_resolved(staged.resolve())
            stale.append(f"{staged} {evidence}")
    if stale:
        raise DecoderContractError(
            "The native Zarr reader rw_zarr found here is older than this woof: "
            + "; ".join(stale) + ". An older reader lacks the coverage query needed "
            "to carry live hydro spin-up past ERA5's published boundary, or refuses "
            "source units such as Hectopascal(hPa). "
            + _remedy(filename))
    raise FileNotFoundError(
        "The native Zarr reader rw_zarr is not installed. " + _remedy(filename))


def extract_regular_zarr(request: dict, *, request_path: Path,
                         output: Path, progress=print) -> dict:
    """Run native acquisition with bounded lifetime and progressive status."""
    executable = resolve_zarr_bin()
    request_path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    report_path = request_path.with_suffix(".result.json")
    with report_path.open("w", encoding="utf-8") as report:
        process = subprocess.Popen([os.fspath(executable), "extract",
            os.fspath(request_path), os.fspath(output)],
            stdout=report, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 24 * 3600
        shown = 0
        try:
            while True:
                try:
                    _, status = process.communicate(timeout=15)
                    done = True
                except subprocess.TimeoutExpired as error:
                    status = error.stderr or ""
                    done = False
                if isinstance(status, bytes):
                    status = status.decode("utf-8", errors="replace")
                complete = status if done else status[:status.rfind("\n") + 1]
                for line in complete[shown:].splitlines():
                    progress(f"fetch: {line}")
                shown = len(complete)
                if done:
                    if process.returncode:
                        raise ValueError(f"Native Zarr acquisition failed: {status[-4000:].strip()}")
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native Zarr acquisition exceeded 24 hours")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if document.get("schema") != "arwen.regular-forcing.v1":
        raise ValueError("Native Zarr reader returned an incompatible output schema")
    return document


@contextmanager
def regular_netcdf_record(path: Path, index: int, variables):
    """Expose one native-decoded time record without rereading the whole window."""
    names = tuple(variables)
    with tempfile.TemporaryDirectory(prefix="arwen-forcing-record-") as temporary:
        root = Path(temporary)
        result = subprocess.run(
            [os.fspath(resolve_zarr_bin()), "dump-record", os.fspath(path),
             str(index), os.fspath(root), *names],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise ValueError(f"Native forcing record decode failed: {result.stderr[-4000:].strip()}")
        document = json.loads(result.stdout)
        if (document.get("schema") != "arwen.regular-forcing-record.v1"
                or document.get("index") != index):
            raise ValueError("Native forcing record has an incompatible schema or time index")
        records = {}
        for item in document.get("variables", []):
            name, leaf = item.get("name"), item.get("file")
            shape = item.get("shape")
            if (name not in names or name in records or not isinstance(leaf, str)
                    or Path(leaf).name != leaf or item.get("dtype") != "<f8"
                    or not isinstance(shape, list)
                    or any(type(size) is not int or size <= 0 for size in shape)):
                raise ValueError("Native forcing record has invalid variable metadata")
            records[name] = (root / leaf, tuple(shape))
        if set(records) != set(names):
            raise ValueError("Native forcing record omitted requested variables")
        yield records
