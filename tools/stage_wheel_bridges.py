"""Stage every declared native artifact into a platform wheel.

Run after the release Cargo builds and before ``python -m build``. The shared
rw_wrfbatch executable and its five native launchers carry the old tool names.
Both package staging and a user's bridge directory receive the full roster.
Basemap assets stay in recast-woof-data and the downloadable bridge bundle.
``tools.battery.wheel_size`` measures wheels before release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from woof import bridge_assets, bridges  # noqa: E402

#: Name of the record this tool writes beside the staged artifacts.  A
#: wheel that carries binaries must be able to SAY which platform they
#: are for and which commit built them, or a mis-tagged wheel is only
#: discoverable by running it.
MANIFEST_NAME = "BUNDLE.json"

#: Schema of that record.
MANIFEST_SCHEMA = "gpuwm-wheel-bridge-bundle-v1"

#: Where each workspace puts its release build, by crate-relative path.
_WORKSPACE_TARGET = {
    "tools/rw_icechunk": "target/release",
    "tools/nextgen_hydrofabric": "target/release",
    "tools/nextgen_verify": "target/release",
    "tools/nextgen_output": "target/release",
    "tools/nextgen_forcing": "target/release",
    "tools/hydro_world": "target/release",
    "tools/hydro_world_params": "target/release",
    "tools/nextgen_reservoir_prepare": "target/release",
    "tools/hydro_world_init": "target/release",
    "tools/ocean_prepare": "target/release",
    "tools/rw-bathy": "target/release",
    "tools/wind-table": "target/release",
    bridges.CRATE_RELATIVE: "target/release",
    bridges.RUSTWX_CRATE_RELATIVE: "target/release",
    "tools/region_global_dealias": "target/release",
    "tools/rw_wps": "target/release",
    "tools/zarr_bridge": "target/release",
}


def purge_build_staging(root: Path | None = None) -> list[Path]:
    """Remove staged artifacts from setuptools' ``build/`` copy tree.

    setuptools copies package data into ``build/lib.<plat>-<pyver>/`` and
    never deletes files that have gone away, so building a second
    platform wheel in the same checkout silently ships the FIRST
    platform's binaries alongside the second's.  Measured, not theorised:
    the linux wheel came out at 105.99 MB carrying 22 artifacts -- every
    Linux binary plus every Windows ``.exe`` and ``.dll`` -- which is
    both wrong and over PyPI's 100 MB cap.

    So every staging operation purges that tree first.  Putting it here
    rather than in a workflow step is deliberate: a release runs this
    tool once per platform, and a cleanup that lives beside the thing it
    cleans cannot be left out of a new caller.
    """

    root = root or _REPO_ROOT
    removed: list[Path] = []
    build = root / "build"
    if not build.is_dir():
        return removed
    for staged in build.glob("lib*/woof/libexec"):
        shutil.rmtree(staged, ignore_errors=True)
        removed.append(staged)
    return removed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(artifact: bridge_assets.BundledArtifact, platform: str) -> Path:
    """Where ``artifact``'s built bytes live in this checkout."""

    target = _WORKSPACE_TARGET.get(artifact.crate)
    if target is None:
        raise SystemExit(
            f"{artifact.name}: crate {artifact.crate!r} has no declared "
            f"release-build directory in stage_wheel_bridges._WORKSPACE_TARGET; "
            f"add it there beside the other workspaces")
    filename = bridge_assets.artifact_filename(artifact, platform)
    return _REPO_ROOT / artifact.crate / target / filename


def _is_packaged_destination(destination: Path) -> bool:
    """Is ``destination`` the in-package directory a wheel carries?

    The two destinations this tool writes have opposite emptying rules,
    and the difference is not cosmetic.  The PACKAGE directory must be
    emptied first: setuptools never removes package data that has gone
    away, so a second platform's staging would ship both platforms'
    binaries in one wheel -- measured at 105.99 MB, over PyPI's cap.
    Any OTHER destination is somebody's directory: a
    ``~/.woof/bridges`` holds the renderer's ``assets/`` basemap tree
    and whatever kept-stale copies its owner parked there, none of which
    this tool staged and none of which it may delete.
    """

    try:
        return destination.resolve() == bridges.packaged_bridge_dir().resolve()
    except OSError:                              # pragma: no cover - exotic FS
        return False


def staged_artifacts(destination: Path) -> tuple[bridge_assets.BundledArtifact, ...]:
    """What ``destination`` takes: the in-package directory gets what the
    wheel carries (:func:`woof.bridge_assets.wheel_artifacts`, which leaves
    out the on-demand artifacts so the wheel stays under PyPI's per-file
    limit); any other destination, a user's bridge directory included, gets
    the whole declared roster, on-demand artifacts and all."""
    if _is_packaged_destination(destination):
        return bridge_assets.wheel_artifacts()
    return bridge_assets.BUNDLED_ARTIFACTS

def stage(platform: str, *, destination: Path | None = None) -> Path:
    """Copy every artifact ``destination`` takes into it, or refuse.

    Fail closed and fail COMPLETE: every missing artifact is collected
    and reported at once, because a build operator who has to re-run
    cargo wants the whole list, not the first name alphabetically.

    The declaration is the whole point.  A directory staged BY HAND has
    nothing to check itself against, which is how a
    ``~/.woof/bridges`` came to hold sixteen of the declared
    artifacts -- missing ``rw_netcdf`` and ``netcdf_writer``, so every
    NetCDF source refused to decode and every history write would have
    refused, on a home that looked staged.  Driven from
    :data:`woof.bridge_assets.BUNDLED_ARTIFACTS`, the artifact that
    joins the declaration next travels to every destination on the same
    commit.
    """

    destination = destination or bridges.packaged_bridge_dir()
    roster = staged_artifacts(destination)
    missing: list[str] = []
    plan: list[tuple[bridge_assets.BundledArtifact, Path, Path]] = []
    for artifact in roster:
        origin = source_path(artifact, platform)
        if not origin.is_file():
            missing.append(f"  {artifact.name:24s} expected at {origin}")
            continue
        plan.append((artifact, origin,
                     destination / bridge_assets.artifact_filename(artifact, platform)))
    if missing:
        raise SystemExit(
            f"stage_wheel_bridges: {len(missing)} of "
            f"{len(roster)} declared artifacts are "
            f"not built for {platform}:\n" + "\n".join(missing) +
            "\n\nBuild them first, from the checkout root:\n"
            + "\n".join(
                "  " + bridges.cargo_build_one_liner(crate)
                for crate in dict.fromkeys(
                    artifact.crate for artifact in roster)))

    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for artifact, origin, target in plan:
        shutil.copy2(origin, target)
        if artifact.kind == "executable" and os.name != "nt":
            target.chmod(target.stat().st_mode | 0o111)
        records.append({
            "artifact": artifact.name,
            "filename": target.name,
            "kind": artifact.kind,
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def stage_from_bundle(archive: Path, platform: str,
                      *, destination: Path | None = None) -> Path:
    """Stage from a packed release bundle instead of a build tree.

    The release builds each platform's binaries on its own target-native
    runner, then one job assembles the distributions.  That job cannot
    have a Windows cargo target directory and a Linux one, so the
    platform wheels are staged from the BUNDLES those runners uploaded --
    the identical bytes the release publishes and pins, which also means
    the wheel and the downloadable bundle can never disagree.

    Only the binaries are taken.  ``assets/`` members are the renderer's
    basemap shapefiles, deliberately not shipped in the wheel (see the
    module docstring).    """

    destination = destination or bridges.packaged_bridge_dir()
    wanted = {
        bridge_assets.artifact_filename(artifact, platform): artifact
        for artifact in staged_artifacts(destination)
    }
    purge_build_staging()
    if _is_packaged_destination(destination) and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        members = set(bundle.namelist())
        missing = sorted(name for name in wanted if name not in members)
        if missing:
            raise SystemExit(
                f"stage_wheel_bridges: {archive} does not carry "
                f"{len(missing)} declared artifact(s) for {platform}: "
                + ", ".join(missing))
        for filename, artifact in sorted(wanted.items()):
            payload = bundle.read(filename)
            target = destination / filename
            target.write_bytes(payload)
            if artifact.kind == "executable" and os.name != "nt":
                target.chmod(target.stat().st_mode | 0o111)
            records.append({
                "artifact": artifact.name,
                "filename": filename,
                "kind": artifact.kind,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "platform": platform,
        "artifacts": records,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _staged_count(destination: Path) -> str:
    """What was staged, as the count the destination's own record holds."""

    staged = len(json.loads(
        (destination / MANIFEST_NAME).read_text(encoding="utf-8"))["artifacts"])
    declared = len(bridge_assets.BUNDLED_ARTIFACTS)
    if staged == declared:
        return f"{staged} artifacts"
    return f"{staged} of {declared} artifacts"

def clean(destination: Path | None = None) -> None:
    purge_build_staging()
    destination = destination or bridges.packaged_bridge_dir()
    if destination.exists():
        shutil.rmtree(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage built Rust artifacts into the package for a "
                    "platform wheel.")
    parser.add_argument(
        "--platform", choices=bridge_assets.SUPPORTED_PLATFORMS,
        default=bridge_assets.host_platform(),
        help="platform key the artifacts were built for (default: this host)")
    parser.add_argument(
        "--from-bundle", type=Path, default=None, metavar="ARCHIVE",
        help="stage from a packed release bundle rather than a build tree "
             "(requires --platform)")
    parser.add_argument(
        "--dest", type=Path, default=None, metavar="DIR",
        help="stage into DIR instead of the in-package directory.  "
             f"{bridges.default_bridge_dir()} is the one woof searches "
             "last, so staging there completes a checkout's own home "
             "from a build tree -- every declared artifact, driven from "
             "the same table the wheel and the release bundle use.  Only "
             "the in-package destination is emptied first; anywhere else "
             "keeps what this tool did not stage")
    parser.add_argument(
        "--clean", action="store_true",
        help="remove the staged directory and exit (restores a pure wheel)")
    args = parser.parse_args(argv)

    if args.clean:
        if args.dest is not None:
            raise SystemExit(
                "--clean removes its destination entirely, so it is "
                "restricted to the in-package directory this tool owns.  "
                f"{args.dest} may hold the renderer's assets/ tree and "
                "copies nobody here staged; delete those deliberately, "
                "not as a flag combination")
        clean()
        print(f"removed {bridges.packaged_bridge_dir()}")
        return 0
    if args.from_bundle is not None:
        if args.platform is None:
            raise SystemExit(
                "--from-bundle needs --platform: a bundle's platform is a "
                "property of the release that built it, not of this host")
        destination = stage_from_bundle(args.from_bundle, args.platform,
                                        destination=args.dest)
        total = sum(path.stat().st_size for path in destination.iterdir()
                    if path.is_file())
        print(f"staged {_staged_count(destination)} for "
              f"{args.platform} from {args.from_bundle} into {destination} "
              f"({total / 1e6:.2f} MB)")
        return 0
    if args.platform is None:
        raise SystemExit(
            "no bundle is published for this host "
            f"({bridge_assets.host_platform_description()}); pass --platform "
            f"explicitly to cross-stage, or build from source.  Known: "
            f"{', '.join(bridge_assets.SUPPORTED_PLATFORMS)}")
    destination = stage(args.platform, destination=args.dest)
    total = sum(path.stat().st_size for path in destination.iterdir()
                if path.is_file())
    print(f"staged {_staged_count(destination)} for "
          f"{args.platform} into {destination} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
