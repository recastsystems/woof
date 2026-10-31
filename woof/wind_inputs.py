"""Wind inventory acquisition and native table decoding, before stepping."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import zipfile


def table_binary() -> Path:
    from woof.bridges import default_bridge_dir, packaged_bridge_dir, executable_name
    name = executable_name("rw-wind-table")
    override = os.environ.get("WOOF_RW_WIND_TABLE")
    candidates = ([Path(override)] if override else [
        packaged_bridge_dir() / name, default_bridge_dir() / name,
        Path(__file__).resolve().parents[1] / "tools/wind-table/target/release" / name])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    from woof.bridges import fetch_on_demand, on_demand_failure
    # On demand since 2.9.11 (PyPI per-file limit): pinned in this release's
    # bundle, staged by the first wind-farm run that needs it.
    if not override and fetch_on_demand("rw-wind-table"):
        for path in candidates:
            if path.is_file():
                return path.resolve()
    failure = on_demand_failure("rw-wind-table")
    raise ValueError("Wind turbine input requires the rw-wind-table decoder"
                     + (f" ({failure})" if failure else "") + ". "
                     "Stage it with woof fetch-bridges --partial --artifact rw-wind-table, or build tools/wind-table with "
                     "cargo build --release --locked and set WOOF_RW_WIND_TABLE.")


def _cache() -> Path:
    root = Path(os.environ.get("WOOF_WIND_CACHE", Path.home() / ".woof/wind"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def inventory_table(year=0) -> Path:
    """Fetch the registry's public inventory once; decode and filter in Rust.

    A commissioning year is a filter, not a reconstruction of retirements,
    curtailment or within-year commissioning dates.
    """
    data = Path(__file__).parent / "data/wind"
    registry = json.loads((data / "databases.json").read_text())
    source = registry["uswtdb"]
    root = _cache()
    csv_path = root / "uswtdb.csv"
    if not csv_path.is_file():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            archive = Path(tmp) / "inventory.zip"
            request = urllib.request.Request(source["download_url"], headers={"User-Agent": "woof"})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            with zipfile.ZipFile(archive) as bundle:
                names = [n for n in bundle.namelist() if n.lower().endswith(".csv")]
                if len(names) != 1:
                    raise ValueError("USWTDB download must contain exactly one CSV inventory")
                prepared = Path(tmp) / "inventory.csv"
                with bundle.open(names[0]) as inp, prepared.open("wb") as out:
                    while block := inp.read(1024 * 1024):
                        out.write(block)
            os.replace(prepared, csv_path)
    digest = hashlib.sha256(csv_path.read_bytes() + (data / "databases.json").read_bytes()
                            + (data / "classes.json").read_bytes()).hexdigest()
    result = root / f"uswtdb-{digest}-{int(year)}.json"
    if not result.is_file():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            output = Path(tmp) / "table.json"
            subprocess.run([str(table_binary()), str(data / "databases.json"), "uswtdb",
                            str(data / "classes.json"), str(csv_path), str(output),
                            "-180", "-90", "180", "90", str(int(year)), str(os.cpu_count() or 1)],
                           check=True, timeout=300, stdout=subprocess.DEVNULL)
            os.replace(output, result)
    return result


def table_bytes(path) -> bytes:
    if str(path) == "uswtdb":
        path = inventory_table()
    source = Path(path)
    raw = source.read_bytes()
    if raw.lstrip().startswith(b"{"):
        return raw
    completed = subprocess.run([str(table_binary()), "--wrf", str(source.resolve())],
                               check=False, capture_output=True, timeout=90)
    if completed.returncode:
        raise ValueError(f"WRF turbine input {source}: {completed.stderr.decode(errors='replace').strip()}")
    return completed.stdout


def prepare_launch(args):
    """Materialize an immutable go config beside its source, preserving paths."""
    selection = getattr(args, "wind_farms", None)
    if selection is None:
        return
    import tomllib
    from woof.config_authority import read_config_authority
    from woof.toml_document import emit_experiment_toml
    authority = read_config_authority(args.config)
    raw = tomllib.loads(authority.payload.decode())
    selected = selection if selection in ("off", "uswtdb") else str(Path(selection).expanduser().resolve())
    # Resolve downloads during preparation, before a forecast holds a card.
    if selected == "uswtdb":
        when = raw.get("experiment", {}).get("start_time")
        selected = str(inventory_table(getattr(when, "year", 0)))
    shared = raw.setdefault("shared", {})
    shared.update(wind_farms=selected, wind_farm_scheme="off", wind_farm_table="")
    for domain in raw.get("domain", []):
        for key in ("wind_farms", "wind_farm_scheme", "wind_farm_table"):
            domain.pop(key, None)
    payload = emit_experiment_toml(raw).encode()
    target = authority.source.with_name(authority.source.stem + ".wind-" + hashlib.sha256(payload).hexdigest()[:16] + ".toml")
    if not target.exists():
        target.write_bytes(payload)
    elif target.read_bytes() != payload:
        raise ValueError(f"wind configuration identity conflict at {target}")
    args.config = target

