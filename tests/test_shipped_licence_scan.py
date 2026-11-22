"""No public distribution carries a non-commercial, no-derivatives, GPL or
UCAR-no-sale licence.

The breakage this prevents.  The public packages (``woof``,
``recast-woof-data``, ``gpuwm-kernels``, ``gpuwm-kernels-startup``) are
Apache-2.0, and third-party content may ride in them only under a
permissive licence (BSD, MIT, Apache-2.0, public domain) or CC-BY /
CC-BY-SA data with its attribution.  A file under CC-BY-NC*, CC-BY-ND*,
GPL/LGPL/AGPL or UCAR's "shall not sell" terms attaches that restriction
to whatever carries it.  It has happened: through 2.7.x the companion tree
held the four CC-BY-NC-SA-4.0 RFMIP reference results
(``rfmip-clear-sky-reference-{lw,sw}-{up,down}.nc``) beside the shipped
RRTMGP tables, and ``woof_data = ["data/**/*"]`` sweeps in whatever sits
in that directory -- so only an exclusion line stood between them and the
wheel, and stale checkouts still hold them on disk today.  Since 2.8.0 the
tests fetch them from upstream (``woof/core/rfmip_upstream.py``) and
``test_package_data_coverage.py::test_no_rfmip_file_is_carried`` pins those
five names.  This file is the general form: it reads the bytes of every
file each distribution can carry, so the NEXT restricted file is caught by
its licence text, not by its name.

How it reads.  A filesystem walk (not ``git ls-files``), because the
package-data globs sweep up untracked files too: a restricted file left in
a checkout reaches the wheel built from it.  Each file is searched for a
small denylist of licence markers: SPDX identifiers, the licences' own
titles and deed URLs, and UCAR's no-sale clause.  Text files get the whole
list; binary files (a NUL in the first 8 KiB) only the long markers, so
that table bytes cannot match a four-letter identifier by chance.  NetCDF
global ``license`` attributes are stored as plain ASCII, so a restricted
data file is caught by its own embedded licence.

Mentions are not grants.  NOTICE files, provenance tables and refusal
messages name restricted licences precisely in order to keep those works
out, and dual-licensed components ("MIT OR Apache-2.0 OR LGPL-2.1") name
the licence WOOF-ArWen does not elect.  Each such file is listed in
``_EXPLAINED`` with the exact markers it may contain and why.  The list
cannot go stale: an entry whose file is gone, or whose marker no longer
appears, fails too, so a reason outlives nothing it was written for.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache",
                        "build", "dist"})

#: The four public distributions and what each can carry, as roots
#: relative to the repository.  ``("dir", "*")`` walks the whole
#: directory; ``("dir", "*.py")`` takes only the top level's modules
#: (packages.find includes ``tools`` and ``docs`` themselves, not their
#: subdirectories); a plain file is a license-file or top-level page.
#: Deliberately at least as wide as the wheels: a wider scan only makes
#: the gate stricter.
_DISTRIBUTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "woof": (
        ("woof", "*"), ("mpas_cycle_bridge", "*"), ("configs", "*"),
        ("tilestream", "**.py"), ("tools", "*.py"), ("tools", "*.sh"),
        ("docs", "*.py"), ("licenses", "*"),
        ("LICENSE", ""), ("NOTICE", ""), ("README.md", ""),
    ),
    "recast-woof-data": (("recast-woof-data", "*"),),
    "gpuwm-kernels": (("gpuwm-kernels", "*"),),
    "gpuwm-kernels-startup": (("gpuwm-kernels-startup", "*"),),
}

#: Subpackages ``[tool.setuptools.packages.find] exclude`` keeps out of
#: the woof wheel.
_EXCLUDED_SUBTREES = ("tilestream/rescued-tools", "tilestream/skeptic")

#: name -> (pattern, applies to binary files too).
_MARKERS: dict[str, tuple[re.Pattern[bytes], bool]] = {
    # SPDX / prose identifiers: CC-BY-NC-SA-4.0, "CC BY-NC 4.0",
    # CC-BY-ND-4.0, CC-BY-NC-ND ...
    "cc-nc-nd-id": (re.compile(rb"CC[- _]?BY[- _]?N[CD]"), True),
    "cc-nc-nd-title": (re.compile(
        rb"Attribution[- ]No(?:nCommercial|Derivatives|Derivs)"), True),
    "cc-nc-nd-deed": (re.compile(
        rb"creativecommons\.org/licenses/by-n[cd]"), True),
    "non-commercial": (re.compile(rb"[Nn]on-?[Cc]ommercial"), True),
    "gpl-title": (re.compile(
        rb"GNU (?:LESSER |AFFERO |LIBRARY )?GENERAL PUBLIC LICENSE"
        rb"|GNU (?:Lesser |Affero |Library )?General Public License"), True),
    "gpl-id": (re.compile(
        rb"\b(?:A|L)?GPL(?:-[123](?:\.[01])?|v[123])\b|\bGPL-[23]\+"), False),
    "ucar-no-sale": (re.compile(rb"shall not sell"), True),
}

_BINARY_PROBE = 8192

#: Files that name a restricted licence without being under one: the exact
#: markers each may contain, and why that is a mention rather than a grant.
_WHY_NOTICE = ("third-party notice: names restricted works WOOF-ArWen does NOT "
               "ship (RFMIP reference results, Lynker hydrofabric v2.2, "
               "UPP's LGPL source) and the permissive licence it elects for "
               "dual-licensed crates")
_WHY_CCCL = ("NVIDIA CCCL is Apache-2.0 WITH LLVM-exception; the exception "
             "text itself names GPLv2 compatibility")
_WHY_CRATES = ("Rust crate licence inventory for the shipped binaries: "
               "r-efi is MIT OR Apache-2.0 OR LGPL-2.1-or-later (MIT "
               "elected), aws-lc names GPLv2 only for its OpenSSL-compat "
               "alternative Amazon does not elect")
_WHY_ZARR = (_WHY_CRATES + "; lz4-sys vendors the liblz4 repository whose "
             "programs/, tests/ and examples/ are GPL-2.0-or-later, while "
             "the crate compiles only lib/ (BSD-2-Clause) into the binary, "
             "and the inventory reproduces every licence text the vendored "
             "source carries")
_WHY_RFMIP_DOC = ("records that the four RFMIP reference results are "
                  "CC-BY-NC-SA-4.0 and therefore fetched from upstream at "
                  "test time, never shipped (woof/core/rfmip_upstream.py)")

_WHY_AER_WRF = ("reproduces the copyright notice WRF's copy of AER RRTMG "
                "carries, as that copy's grant requires, and states that "
                "WOOF-ArWen takes the code under AER's current BSD-3-Clause "
                "grant, not the older non-commercial terms it quotes")
_WHY_GPL_TEXT = ("the GPL text reproduced for the GPL-2.0 half of "
                 "dual-licensed crates (self_cell Apache-2.0 OR GPL-2.0, "
                 "zstd BSD OR GPL-2.0); the permissive half is elected")

#: (WOOF 1.1.7) the hex model's files, each read in full for the 1.1.7 seal.
_WHY_NOTICE_HEX = ("; its hex part (section 4) quotes UCAR's Noah-MP v5.0 "
                   "'shall not sell' clause to say why that port is withheld "
                   "from this distribution, and it is")
_WHY_HEX_HELPERS = ("the hex helpers' crate inventory (286 crates, every grant "
                    "permissive and the permissive branch taken); the GPLv2 "
                    "tokens are the Apache-2.0 LLVM exception text that "
                    "linux-raw-sys, rustix and wasi offer and that the "
                    "inventory reproduces")
_WHY_RRFS_SKIP = ("the RRFS pin set's source manifest records that the fork's "
                  "stochastic_physics submodule is LGPL-3.0 and lists it under "
                  "run_time_skip, so woof.hex.mpasphys.resources_build never "
                  "fetches it; its file rows are hashes for the referee build's "
                  "source check, not the files")
_EXPLAINED: dict[str, tuple[frozenset[str], str]] = {
    "NOTICE": (frozenset({"cc-nc-nd-id", "gpl-title", "gpl-id",
                          "non-commercial", "ucar-no-sale"}),
               _WHY_NOTICE + _WHY_NOTICE_HEX),
    "gpuwm-kernels/NOTICE": (
        frozenset({"cc-nc-nd-id", "gpl-title", "gpl-id", "non-commercial"}),
        _WHY_NOTICE),
    "gpuwm-kernels-startup/NOTICE": (
        frozenset({"cc-nc-nd-id", "gpl-title", "gpl-id", "non-commercial"}),
        _WHY_NOTICE),
    "recast-woof-data/NOTICE": (frozenset({"cc-nc-nd-id", "non-commercial"}),
                          _WHY_RFMIP_DOC),
    "recast-woof-data/README.md": (frozenset({"cc-nc-nd-id"}), _WHY_RFMIP_DOC),
    "recast-woof-data/MANIFEST.in": (frozenset({"cc-nc-nd-id"}), _WHY_RFMIP_DOC),
    "recast-woof-data/pyproject.toml": (frozenset({"cc-nc-nd-id"}),
                                  _WHY_RFMIP_DOC),
    "recast-woof-data/woof_data/data/rrtmgp/PROVENANCE.md": (
        frozenset({"cc-nc-nd-id", "non-commercial"}), _WHY_RFMIP_DOC),
    "woof/data/wrf_radiation/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt": (
        frozenset({"non-commercial"}), _WHY_AER_WRF),
    "woof/data/cam3/RUST-NOTICES.txt": (
        frozenset({"non-commercial"}),
        "a permissive crate licence granting use 'for any purpose, "
        "commercial or non-commercial'"),
    "tools/hf_inventory.py": (
        frozenset({"non-commercial"}),
        "the model-card classifier: marks NC/ND or non-commercial cards "
        "research-only so they are refused"),
    "woof/core/rfmip_upstream.py": (
        frozenset({"cc-nc-nd-id", "non-commercial"}),
        "the on-demand fetcher: pins each upstream RFMIP file with the "
        "licence it is published under, which is why it is not shipped"),
    "licenses/LICENSE-CCCL.txt": (frozenset({"gpl-id"}), _WHY_CCCL),
    # (WOOF 1.1.7) the hex model's copy of the same CCCL text, byte for byte
    # (libcu++ cuda::barrier in the RRFS smoke workers; hex NOTICE section 16)
    "licenses/LICENSE.libcudacxx": (frozenset({"gpl-id"}), _WHY_CCCL),
    "licenses/THIRD-PARTY-LICENSES-helper-binaries.txt": (
        frozenset({"gpl-id"}), _WHY_HEX_HELPERS),
    "woof/hex/data/mpasphys-native/mpas-source-rrfs-44c0849.json": (
        frozenset({"gpl-id"}), _WHY_RRFS_SKIP),
    "woof/hex/mpasphys/resources_build.py": (
        frozenset({"gpl-id"}),
        "the comment at the run_time_skip filter that keeps the LGPL-3.0 "
        "stochastic_physics Registry out of every fetch"),
    "woof/hex/data/profiles/licence-gates.v1.json": (
        frozenset({"ucar-no-sale"}),
        "the licence gate's table: it quotes UCAR's Noah-MP clause so the "
        "forecast door refuses a Noah-MP v5.0 selection unless the run names "
        "--licence-opt-in ucar-noahmp; no default selects one"),
    "woof/hex/earth2_init_rows.json": (
        frozenset({"cc-nc-nd-id"}),
        "the Earth-2 start rows record each model's weight licence; the four "
        "NC/ND rows (Pangu and GraphCast x2 CC BY-NC-SA, FuXi-S2S CC BY-NC-ND) "
        "are default_enabled false and refused unless the exact licence is "
        "named, and no weights are carried"),
    "gpuwm-kernels/licenses/LICENSE-CCCL.txt": (
        frozenset({"gpl-id"}), _WHY_CCCL),
    "gpuwm-kernels-startup/licenses/LICENSE-CCCL.txt": (
        frozenset({"gpl-id"}), _WHY_CCCL),
}
for _root in ("licenses", "gpuwm-kernels/licenses",
              "gpuwm-kernels-startup/licenses"):
    _EXPLAINED[f"{_root}/RW-ICECHUNK-THIRD-PARTY.txt"] = (
        frozenset({"gpl-title", "gpl-id"}), _WHY_CRATES)
    _EXPLAINED[f"{_root}/THIRD-PARTY-LICENSES-bridge-binaries.txt"] = (
        frozenset({"gpl-title", "gpl-id", "non-commercial"}),
        _WHY_CRATES + "; " + _WHY_GPL_TEXT + " (the GPL text itself says "
        "'noncommercial distribution')")
    _EXPLAINED[f"{_root}/THIRD-PARTY-LICENSES-zarr-binary.txt"] = (
        frozenset({"gpl-title", "gpl-id", "non-commercial"}),
        _WHY_ZARR + " (the GPL text itself says 'noncommercial "
        "distribution')")
    _EXPLAINED[f"{_root}/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt"] = (
        frozenset({"non-commercial"}), _WHY_AER_WRF)
_EXPLAINED.update({
    "woof/authorities/earth2-models.v1.json": (
        frozenset({"cc-nc-nd-id"}),
        "Earth-2 model authority: records each model card's licence so "
        "NC/ND weights are refused or kept owner-only, never downloaded "
        "by a default path"),
    "woof/authorities/hf-earth2-inventory.v1.json": (
        frozenset({"non-commercial"}),
        "inventory of upstream model cards, with the non-commercial ones "
        "marked so the loader refuses them"),
    "woof/authorities/rw-wps-fetch-routes.v1.json": (
        frozenset({"cc-nc-nd-id"}),
        "fetch-route authority: TIGGE is CC-BY-NC-4.0 and named so it is "
        "never an automatic fallback"),
    "woof/source_adapters.py": (
        frozenset({"cc-nc-nd-id"}),
        "the same TIGGE refusal, in the source adapter"),
    "woof/data/hydro/configurations.v1.json": (
        frozenset({"cc-nc-nd-id", "non-commercial"}),
        "hydro fabric authority: the NC-licensed Lynker/NextGen v2.2 "
        "fabric is named research-only and never shipped"),
    "woof/data/hydro/fabric-providers.v1.json": (
        frozenset({"gpl-id"}),
        "py7zr (LGPL-2.1-or-later) is a pip dependency of the opt-in "
        "[hydro-conus] extra, named with its licence; no py7zr code ships"),
    "woof/doctor.py": (
        frozenset({"gpl-id"}),
        "doctor reports whether the opt-in LGPL py7zr extra is installed"),
    "woof/hydro/acquire.py": (
        frozenset({"gpl-id"}),
        "the hydro acquire step names the opt-in LGPL py7zr extra in its "
        "install hint"),
})


def _is_excluded(relative: str) -> bool:
    return any(relative == sub or relative.startswith(sub + "/")
               for sub in _EXCLUDED_SUBTREES)


def _walk(root: Path, pattern: str) -> list[Path]:
    if pattern == "":
        return [root] if root.is_file() else []
    if not root.is_dir():
        return []
    if pattern in ("*.py", "*.sh"):
        return sorted(p for p in root.glob(pattern) if p.is_file())
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS
                             and not d.endswith(".egg-info"))
        for name in filenames:
            if pattern == "**.py" and not name.endswith(".py"):
                continue
            files.append(Path(dirpath, name))
    return files


def _carried_files() -> dict[str, set[str]]:
    """Repo-relative path -> the distributions that can carry it."""

    carried: dict[str, set[str]] = {}
    for distribution, roots in _DISTRIBUTIONS.items():
        for root, pattern in roots:
            for path in _walk(REPO_ROOT / root, pattern):
                relative = path.relative_to(REPO_ROOT).as_posix()
                if _is_excluded(relative):
                    continue
                carried.setdefault(relative, set()).add(distribution)
    return carried


def _markers_in(path: Path) -> set[str]:
    data = path.read_bytes()
    binary = b"\0" in data[:_BINARY_PROBE]
    return {name for name, (pattern, in_binary) in _MARKERS.items()
            if (in_binary or not binary) and pattern.search(data)}


def _scan() -> dict[str, set[str]]:
    hits: dict[str, set[str]] = {}
    for relative in sorted(_carried_files()):
        found = _markers_in(REPO_ROOT / relative)
        if found:
            hits[relative] = found
    return hits


def _require_source_tree() -> None:
    if not (REPO_ROOT / "pyproject.toml").is_file() or not (
            REPO_ROOT / "recast-woof-data" / "pyproject.toml").is_file():
        pytest.skip("licence scan needs the source tree, not an install")


@pytest.fixture(scope="module")
def scan() -> dict[str, set[str]]:
    _require_source_tree()
    return _scan()


def test_the_scan_sees_every_distribution() -> None:
    """A scan that walks nothing reports green; this test makes it walk."""

    _require_source_tree()
    carried = _carried_files()
    seen = {d for owners in carried.values() for d in owners}
    assert seen == set(_DISTRIBUTIONS), sorted(set(_DISTRIBUTIONS) - seen)
    # The directory the RFMIP reference results used to sit in is walked.
    assert any(p.startswith("recast-woof-data/woof_data/data/rrtmgp/")
               for p in carried)


def test_no_carried_file_is_under_a_restricted_licence(scan) -> None:
    """Every licence marker in a carried file is an explained mention."""

    unexplained = {}
    for relative, found in scan.items():
        allowed = _EXPLAINED.get(relative, (frozenset(), ""))[0]
        extra = found - allowed
        if extra:
            unexplained[relative] = sorted(extra)
    assert not unexplained, (
        "these files can ship in a public Apache-2.0 distribution and carry "
        "a non-commercial / no-derivatives / GPL / UCAR-no-sale licence "
        "marker. If the file is under that licence it must leave the "
        "package (fetch it on demand, or replace it with a permissive "
        "source); if it only names the licence, add it to _EXPLAINED with "
        f"the reason: {unexplained}")


def test_every_explanation_still_applies(scan) -> None:
    """A reason whose file or marker is gone is retired, not kept."""

    stale = {}
    for relative, (allowed, _why) in _EXPLAINED.items():
        missing = allowed - scan.get(relative, set())
        if missing:
            stale[relative] = sorted(missing)
    assert not stale, f"_EXPLAINED entries no longer needed: {stale}"


def test_the_markers_catch_the_files_this_gate_exists_for(tmp_path) -> None:
    """The RFMIP reference results' own licence text trips the scan, and a
    permissive one does not."""

    restricted = tmp_path / "ref.nc"
    # The global attribute the four RFMIP reference files carry.
    restricted.write_bytes(
        b"CDF\x01\x00\x00\x00\x00license\x00Creative Commons "
        b"Attribution-NonCommercial-ShareAlike 4.0 International "
        b"(CC BY-NC-SA 4.0)")
    assert {"cc-nc-nd-id", "cc-nc-nd-title", "non-commercial"} <= (
        _markers_in(restricted))
    for text in (b"SPDX-License-Identifier: GPL-3.0-only",
                 b"GNU GENERAL PUBLIC LICENSE\n Version 2",
                 b"Licensee shall not sell, rent, lease"):
        path = tmp_path / "f.txt"
        path.write_bytes(text)
        assert _markers_in(path), text
    permissive = tmp_path / "ok.txt"
    permissive.write_bytes(
        b"BSD 3-Clause License\nApache-2.0\nCC-BY-SA-4.0\nCC-BY-4.0\nMIT\n")
    assert not _markers_in(permissive)
    # A short identifier is not searched in table bytes.
    table = tmp_path / "t.dat"
    table.write_bytes(b"\0\x01GPLv3\x02")
    assert not _markers_in(table)
