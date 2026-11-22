"""Build the canonical fork tables from hash-pinned public source on one CPU.

Only the small original Fortran harness ships. Generated coefficients must
match every existing fork-table pin; compiler differences never create a new
accepted numerical setup. Linux x86-64 needs GNU Fortran and dpkg-deb. Windows
uses an installed WSL Ubuntu distribution with those CPU tools.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from woof.core.thompson_contract import TableAsset, FORK_TABLE_ASSETS, validate_table_assets
from woof.table_assets import fetch_asset_from_url, TableAssetError

SOURCE_BASE = "https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21/sorc/hrrr_wrfarw.fd/WRFV3.9/phys"
PUBLIC_SOURCES = (
    TableAsset("module_mp_thompson.F", 218475,
               "4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283"),
    TableAsset("module_mp_radar.F", 24549,
               "08329c87604b234efab53f7986163a4f6050f58823da9099cb464163c1920f08"),
)
LIBC_ASSET = TableAsset("libc6_2.43-2ubuntu2.4_amd64.deb", 2104056,
    "a613b457f3ff9c84ebd28af8a05afdec22605ad9aed87c71e3a6ae5170aeb9ca")
# The Launchpad librarian object, which keeps every published Ubuntu binary;
# the archive pool drops a version once it is superseded (see
# thompson_density_tables.REFERENCE_RUNTIME_ASSETS).
LIBC_URL = "https://launchpadlibrarian.net/875753646/" + LIBC_ASSET.filename
HARNESS_PINS = {
    "stub_wrf.F90": "e377d142027d294cb5d259f02c2c04ef68d698ff98816ba04de263af003396ba",
    "generate_tables.F90": "65a0925b31003ae24135e1a0c69c48ccc08e03b0195c6a3c3282c3a5d0c03f77",
}
WSL_DISTRO_ENV = "WOOF_THOMPSON_FORK_WSL_DISTRO"

BUILD_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
work=$1
cd "$work"
command -v gfortran >/dev/null || { echo 'GNU Fortran (gfortran) is required'; exit 2; }
command -v dpkg-deb >/dev/null || { echo 'dpkg-deb is required for the pinned portable libc'; exit 2; }
test "$(uname -m)" = x86_64 || { echo 'The pinned portable libc needs Linux x86_64'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flags=(-O2 -fno-tree-vectorize -ffree-form -ffree-line-length-none)
gfortran -c "${flags[@]}" stub_wrf.F90
gfortran -c "${flags[@]}" sources/module_mp_radar.F
gfortran -c "${flags[@]}" -cpp -DWRF_CHEM=0 sources/module_mp_thompson.F -o module_mp_thompson.o
gfortran -c "${flags[@]}" generate_tables.F90
gfortran "${flags[@]}" -o generate_tables stub_wrf.o module_mp_radar.o module_mp_thompson.o generate_tables.o
if nm -D generate_tables | grep -q _ZGV; then echo 'Refused vector libm generation'; exit 2; fi
dpkg-deb -x "sources/libc6_2.43-2ubuntu2.4_amd64.deb" libc
lib="$work/libc/usr/lib/x86_64-linux-gnu"
loader="$lib/ld-linux-x86-64.so.2"
printf 'compiler: '; gfortran --version | head -1
"$loader" --library-path "$lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu" --list "$work/generate_tables"
mkdir generated
cd generated
nice -n 15 timeout 300s "$loader" --library-path "$lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu" "$work/generate_tables"
sha256sum qr_acr_qg.dat qr_acr_qs.dat freezeH2O.dat thompson_aux_tables.dat
'''


def _command(script: Path, work: Path) -> list[str]:
    if os.name == "nt":
        distro = os.environ.get(WSL_DISTRO_ENV, "Ubuntu-24.04")
        if shutil.which("wsl.exe") is None:
            raise FileNotFoundError("Canonical fork generation needs WSL Ubuntu with gfortran and dpkg-deb, or a pinned --from directory/mirror")
        prefix = ["wsl.exe", "-d", distro, "--"]
        convert = lambda path: subprocess.check_output(
            [*prefix, "wslpath", "-a", "-u", str(path.resolve()).replace("\\", "/")], text=True).strip()
        return [*prefix, "bash", convert(script), convert(work)]
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise FileNotFoundError("Canonical fork generation needs Linux x86_64 with gfortran and dpkg-deb, or a pinned --from directory/mirror")
    if shutil.which("gfortran") is None or shutil.which("dpkg-deb") is None:
        raise FileNotFoundError("Canonical fork generation needs GNU Fortran (gfortran) and dpkg-deb, or a pinned --from directory/mirror")
    return ["bash", str(script), str(work)]


def build_canonical_fork_tables(work: Path, *, log_path: Path) -> Path:
    """Acquire pinned source, execute unchanged Fortran, and verify all outputs.

    The caller owns this unique work directory, its root lock and cleanup.
    No system package, home cache, GPU, or external build directory is used.
    """
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    script = work / "build.sh"
    script.write_text(BUILD_SCRIPT, encoding="utf-8", newline="\n")
    try:
        command = _command(script, work)
    except (OSError, subprocess.SubprocessError) as error:
        raise TableAssetError(str(error)) from error
    source = work / "sources"
    source.mkdir()
    for asset in PUBLIC_SOURCES:
        fetch_asset_from_url(source, asset, SOURCE_BASE + "/" + asset.filename)
    fetch_asset_from_url(source, LIBC_ASSET, LIBC_URL)
    harness = Path(__file__).parent / "data" / "thompson" / "fork-build"
    for name, expected in HARNESS_PINS.items():
        original = harness / name
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise TableAssetError(f"Packaged fork generation harness {original} differs from its pinned source")
        shutil.copyfile(original, work / name)
    print("woof fork tables: building the pinned Fortran source on one CPU; "
          "first use can take several minutes", flush=True)
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=330,
                              env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    except (OSError, subprocess.SubprocessError) as error:
        raise TableAssetError(f"Canonical fork table build failed: {error}") from error
    log_path.write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode:
        raise TableAssetError(f"Canonical fork table build exited {done.returncode}; see {log_path}. Install GNU Fortran and dpkg-deb in Linux/WSL or supply a pinned --from set.")
    generated = work / "generated"
    try:
        validate_table_assets(generated, FORK_TABLE_ASSETS)
    except (OSError, ValueError) as error:
        raise TableAssetError(f"Generated fork tables do not match the canonical manifest; refused without changing any pin: {error}; see {log_path}") from error
    receipt = {"schema": 1, "command": command,
               "public_sources": [{"filename": a.filename, "sha256": a.sha256, "bytes": a.bytes} for a in (*PUBLIC_SOURCES, LIBC_ASSET)],
               "harness_sha256": HARNESS_PINS, "build_script_sha256": hashlib.sha256(BUILD_SCRIPT.encode()).hexdigest(),
               "assets": [{"filename": a.filename, "sha256": a.sha256, "bytes": a.bytes} for a in FORK_TABLE_ASSETS]}
    (work / "source-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return generated
