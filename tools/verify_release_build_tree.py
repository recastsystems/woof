"""Refuse a release build whose tree is not "the tagged commit plus the pin step".

Two documents about the same bytes are true at once, by design:

* the COMMITTED ``woof/data/bridges/bridge-pins.json`` at a release tag says
  ``release: null, platforms: {}`` -- GitHub's automatic source archives are
  snapshots of the tagged tree and must not impersonate pinned release bytes
  (``tools/verify_source_bridge_pins.py``, a hard gate by owner ruling
  2026-08-03, and the ``cut`` job of ``.github/workflows/publish.yml``);
* the WORKING COPY of the same file, in the build workspace, is pinned by
  ``tools/build_bridge_bundle.py pin`` from the exact bundle bytes just
  built, and the wheel and sdist are built from that working copy
  (``setup.py`` refuses a wheel from an unpinned tree).

So the pin step is the ONE working-tree mutation a release build is allowed
to carry, and nothing checked that it was the only one.  The 2.7.0 candidate
showed what that costs: the engine wheels were built correctly from
c6391a83b plus pins, but the SDK kit beside them was generated from a tree
with 48 dirty files (XPORT-002), and a reviewer reading ``git status`` in
the build tree could not tell the two situations apart (PKGENG-003, RC2-003).

This gate names the three breakages and refuses each by name:

1. any tracked modification, addition or deletion other than the pins file,
   or any untracked file under a packaged root -- the artifacts would carry
   bytes the tagged commit does not, so the release cannot be reproduced
   from its tag;
2. a release version whose working pins are still ``release: null`` -- the
   2.5.0 blocker verbatim (``pip install recast-woof && woof setup`` reports
   FAILED bridges on a clean home), caught here BEFORE the wheel build
   rather than by setup.py after the multi-hour bundle builds;
3. a COMMITTED pins document that is pinned -- it would fail
   ``verify_source_bridge_pins.py`` at the tag and make GitHub's source
   archive claim bytes it does not carry.

Run it in the build workspace immediately before ``python -m build`` (the
CI ``prepare`` job after its pin step, and the offline desktop build alike):

    python tools/verify_release_build_tree.py

A pre-release or development version (``2.7.0rc1``, ``2.7.0.dev3``, a local
``+`` suffix) is allowed unpinned working pins, because a dev wheel built
under ``WOOF_ALLOW_UNPINNED_WHEEL=1`` is exactly what those trees make;
the cleanliness and committed-pins checks still apply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS_RELATIVE = "woof/data/bridges/bridge-pins.json"

#: Top-level directories whose files reach the wheel or sdist through
#: ``[tool.setuptools.packages.find]`` and ``package-data`` globs, so an
#: UNTRACKED file under them ships.  Everything else untracked (a stray log
#: at the root, a scratch directory) stays out of the artifacts and is not
#: this gate's business.
PACKAGED_ROOTS = ("woof", "tools", "tilestream", "configs", "docs",
                  "mpas_cycle_bridge", "recast-woof-data", "gpuwm-kernels", "gpuwm-kernels-startup")

#: The two platforms a release pins; kept in step with
#: ``woof.bridge_assets.SUPPORTED_PLATFORMS`` by the test, not by import,
#: because this tool runs in build workspaces where woof may not import.
SUPPORTED_PLATFORMS = ("linux-x86_64", "win-x86_64")

_RELEASE_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class ReleaseBuildTreeError(ValueError):
    """The tree is not the tagged commit plus the pin step."""


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise ReleaseBuildTreeError(
            f"git {' '.join(args)} failed in {repo}: {done.stderr.strip()}")
    return done.stdout


def declared_version(repo: Path) -> str:
    with (repo / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def is_release_version(version: str) -> bool:
    """A plain ``X.Y.Z``; anything with a pre, dev, post or local part is not."""

    return bool(_RELEASE_VERSION.match(version.strip()))


def _load_pins(text: str, what: str) -> dict:
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise ReleaseBuildTreeError(f"{what} pins document is not JSON: {error}")
    if not isinstance(payload, dict):
        raise ReleaseBuildTreeError(f"{what} pins document is not a JSON object")
    return payload


def working_tree_changes(repo: Path) -> list[str]:
    """``git status --porcelain`` rows, minus the one permitted mutation."""

    rows = [row for row in _git(repo, "status", "--porcelain",
                                "--untracked-files=all").splitlines()
            if row.strip()]
    offending: list[str] = []
    for row in rows:
        code, path = row[:2], row[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"').replace("\\", "/")
        if code == "??":
            if path.split("/", 1)[0] in PACKAGED_ROOTS:
                offending.append(f"{row}  (untracked, under a packaged root)")
            continue
        if path == PINS_RELATIVE and code.strip() in ("M", "AM", "MM"):
            continue                    # the pin step, the one allowed change
        offending.append(row)
    return offending


def verify_build_tree(repo: Path = REPO_ROOT, *, version: str | None = None
                      ) -> dict:
    """Return a receipt or raise :class:`ReleaseBuildTreeError` naming the breakage."""

    repo = Path(repo).resolve()
    version = declared_version(repo) if version is None else str(version)
    release = is_release_version(version)
    receipt = {"repo": str(repo), "version": version,
               "release_version": release}

    offending = working_tree_changes(repo)
    if offending:
        raise ReleaseBuildTreeError(
            "the build tree carries changes beyond the pin step, so the "
            "wheel and sdist built from it cannot be reproduced from the "
            "tagged commit (the 2.7.0 rc2 SDK kit came from a tree with 48 "
            f"dirty files).  Commit or discard each of these first:\n  "
            + "\n  ".join(offending)
            + f"\nOnly {PINS_RELATIVE} may differ, and only as the output of "
            "`python tools/build_bridge_bundle.py pin`.")

    committed = _load_pins(_git(repo, "show", f"HEAD:{PINS_RELATIVE}"),
                           "committed")
    if committed.get("release") is not None or committed.get("platforms"):
        raise ReleaseBuildTreeError(
            f"HEAD's {PINS_RELATIVE} is PINNED (release="
            f"{committed.get('release')!r}).  A tagged tree must carry "
            "release: null and platforms: {}: GitHub's automatic source "
            "archive is a snapshot of the tag and would claim bundle bytes "
            "it does not carry (tools/verify_source_bridge_pins.py, owner "
            "ruling 2026-08-03; publish.yml's cut job runs it and refuses).  "
            "Pins belong in the build workspace's working copy only; do not "
            "commit them.")
    receipt["committed_pins"] = "unpinned"

    working = _load_pins((repo / PINS_RELATIVE).read_text(encoding="utf-8"),
                         "working-copy")
    pinned = bool(working.get("release")) and bool(working.get("platforms"))
    receipt["working_pins_release"] = working.get("release")
    receipt["working_pins_platforms"] = sorted(working.get("platforms") or {})
    if release:
        expected = f"v{version}"
        if not pinned:
            raise ReleaseBuildTreeError(
                f"version {version} is a release and the working-copy "
                f"{PINS_RELATIVE} still declares release: null.  A wheel "
                "built now is the 2.5.0 blocker verbatim: on a clean home "
                "`pip install recast-woof && woof setup` reports FAILED bridges.  "
                "Run the pin step first:\n  python tools/build_bridge_bundle.py "
                f"pin --release {expected} --source-rev <commit> --bundle "
                "<each bundle from `pack`> --out " + PINS_RELATIVE)
        if working.get("release") != expected:
            raise ReleaseBuildTreeError(
                f"the working-copy pins declare release {working.get('release')!r} "
                f"but pyproject.toml declares version {version}; the wheel "
                f"would fetch {working.get('release')!r}'s bundles under "
                f"{expected}'s name.")
        missing = sorted(set(SUPPORTED_PLATFORMS) - set(working["platforms"]))
        if missing:
            raise ReleaseBuildTreeError(
                f"the working-copy pins lack platform(s) {missing}; a release "
                "pins every platform in woof.bridge_assets.SUPPORTED_PLATFORMS "
                "or `woof setup` on that platform refuses with 'no bundle "
                "pinned'.")
        receipt["verdict"] = "release build tree: clean, committed unpinned, working pinned"
    else:
        receipt["verdict"] = ("pre-release/dev build tree: clean, committed "
                              "unpinned; working pins "
                              + ("pinned" if pinned else "unpinned (allowed)"))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT,
                        help="repository to check (default: this checkout)")
    parser.add_argument("--version", default=None,
                        help="override the version read from pyproject.toml")
    args = parser.parse_args(argv)
    try:
        receipt = verify_build_tree(args.repo, version=args.version)
    except ReleaseBuildTreeError as error:
        print(f"verify_release_build_tree: REFUSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
