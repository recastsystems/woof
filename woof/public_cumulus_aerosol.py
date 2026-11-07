"""Acquire the public monthly prescribed aerosol input; Rust owns conversion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

FILENAME = "NASA_MERRA2_GMI_2001_2010_Aerosol.dat"
SOURCE = Path(__file__).parent / "data/cumulus_aerosol/public-source.json"
LABEL = "written from papers, not bitwise to WRF"


def digest(path):
    # Metadata timestamps can collide on shared filesystems. Validate the
    # actual bytes at each use rather than authorizing cached content by stat.
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def binary(name, variable):
    from woof.bridges import executable_name, find_artifact
    path = find_artifact(variable, executable_name(name), crate_relative="tools/aerosol_grid")
    if path is not None:
        return path
    # Both small providers belong to the platform payload. Source checkouts
    # can build them without downloading a toolchain or running cargo here.
    raise RuntimeError(f"{name} is absent: build tools/aerosol_grid with cargo build --release --locked --offline, or set {variable}; monthly aerosol preparation cannot run without the Rust provider")


def public_asset(source_directory):
    """Default on-demand public replacement; never open the uncleared table."""
    source = json.loads(SOURCE.read_text())
    root = Path(os.environ.get("WOOF_PUBLIC_AEROSOL_CACHE", Path.home() / ".woof/aerosol"))
    override = os.environ.get("WOOF_PUBLIC_AEROSOL")
    candidates = [Path(override)] if override else [Path(source_directory) / FILENAME, root / FILENAME]
    asset = next((p for p in candidates if p.is_file()), None)
    if override and asset is None:
        raise FileNotFoundError(f"WOOF_PUBLIC_AEROSOL names a missing public climatology: {override}")
    if asset is None:
        root.mkdir(parents=True, exist_ok=True)
        builder = binary("public-aerosol-build", "GPUWM_PUBLIC_AEROSOL_BUILD")
        print("Preparing NASA MERRA-2 GMI 2001–2010 monthly aerosol climatology; " + LABEL + ". The first preparation downloads public data; later runs reuse the checked file.", file=sys.stderr, flush=True)
        # Each builder owns its temporary output. Only a validated complete
        # record file is published, so interrupted/concurrent runs cannot leave
        # a partial forcing under the final name.
        with tempfile.TemporaryDirectory(prefix="build-", dir=root) as work:
            staged = Path(work) / FILENAME
            subprocess.run([str(builder), str(root / "source"), str(staged), str(source["source_grid_stride"])], check=True)
            _validate(staged, source)
            os.replace(staged, root / FILENAME)
        asset = root / FILENAME
    sha256 = _validate(asset, source)
    receipt = {**source, "filename": FILENAME, "sha256": sha256, "bytes": asset.stat().st_size, "label": LABEL}
    print("Monthly aerosol forcing: " + json.dumps(receipt, sort_keys=True), file=sys.stderr, flush=True)
    return asset, receipt


def _validate(path, source):
    expected = source.get("output_sha256")
    if not expected:
        raise ValueError("Public aerosol source has no generated output pin; an unqualified or different species conversion cannot be used as the default forcing")
    actual = digest(path)
    if path.stat().st_size != source["output_bytes"] or actual != expected:
        raise ValueError(f"Public aerosol input {path} differs from the pinned NASA conversion; rebuilding from the declared public source prevents using a corrupt or differently mapped forcing")
    return actual
