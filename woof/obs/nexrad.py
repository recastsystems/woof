"""Locate and drive the vendored NEXRAD front door (``rw_nexrad``).

Same shape as :mod:`woof.rustwx_fetch`: the Rust bin moves bytes and
reports facts, Python decides what they mean, and the binary is *pointed
at* rather than built, through the resolution ladder
:mod:`woof.bridges` defines:

1. the ``WOOF_RW_NEXRAD`` environment variable naming the built file
   (a missing file it names is a hard error, never silently skipped);
2. a source checkout's ``tools/rustwx/target/{release,debug}``;
3. ``<root>/libexec/bridges`` beside the package;
4. ``~/.woof/bridges``.

Nothing here runs cargo.  Resolution has one side effect and one
only: an artifact found in ``~/.woof/bridges`` that is not the one this
release pinned is re-fetched before it is handed to a door
(:func:`woof.bridges.require_release_pin`).  ``woof doctor`` resolves
inside :func:`woof.bridges.inspection_only`, where there is no side
effect at all, so the report still says what the estate IS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir,
                           legacy_bridge_candidates, lazy_build_hints,
                           rustwx_build_hint,
                           accept_resolved, executable_name,
                           packaged_bridge_dir)

#: Environment variable naming a prebuilt NEXRAD front door.
NEXRAD_ENV = "WOOF_RW_NEXRAD"

#: Executable base name.
NEXRAD_NAME = "rw_nexrad"

#: Schemas the four subcommands print.  A record that does not declare the
#: expected one is a mismatched binary, not a parse problem.
LIST_SCHEMA = "gpuwm-obs.nexrad-list.v1"
FETCH_SCHEMA = "gpuwm-obs.nexrad-fetch.v1"
DECODE_SCHEMA = "gpuwm-obs.nexrad-decode.v1"
VERIFY_SCHEMA = "gpuwm-obs.nexrad-verify.v1"
SITES_SCHEMA = "gpuwm-obs.nexrad-sites.v1"
LIVE_LIST_SCHEMA = "gpuwm-obs.nexrad-live-list.v1"
LIVE_FETCH_SCHEMA = "gpuwm-obs.nexrad-live-fetch.v1"

#: The two acquisition routes, under the names their records publish.  A
#: receipt says which feed served it with one of these, so "where did this
#: observation come from" is a recorded value rather than a bucket name a
#: reader has to recognise.
ARCHIVE_FEED = "archive-volumes"
LIVE_FEED = "live-chunks"

#: The exact ``--abi`` line this wrapper was written against.
#:
#: The live half is appended rather than folded in: a wrapper that drives
#: the real-time route needs a binary that has it, and a binary predating
#: it fails the probe instead of failing at the first ``live-fetch``.
NEXRAD_ABI_MARKER = (
    "gpuwm-obs.nexrad-fetch.v1\tsite\twindow\tbucket\tvolumes\tbytes\t"
    "cache_hits\tsha256\tgpuwm-obs.radar-sweeps.v1\t"
    "gpuwm-obs.nexrad-live-fetch.v1\tfeed\tvolume_id\tchunks\tcomplete\t"
    "lag_seconds")

#: The bucket a command with no ``--bucket`` actually reads from.
#:
#: This must be, and is, the value ``rw-nexrad`` declares as its own
#: ``DEFAULT_BUCKET`` (``tools/rustwx/crates/rw-nexrad/src/s3.rs``): the
#: front door chooses the bucket, this wrapper passes ``--bucket`` only
#: when a caller names one, and a Python constant naming a different
#: bucket is a statement about the pipeline that the pipeline does not
#: make.  It published ``noaa-nexrad-level2`` -- the archive of record,
#: which on 2026-07-30 answered anonymous ListObjectsV2 with 403 -- long
#: after the Rust default was moved to the mirror on a capability check,
#: so an operator reading the Python surface was told the pipeline uses a
#: bucket it cannot reach.  ``tests/test_obs_nexrad.py`` binds this to
#: the built binary's own declared default rather than to a second copy
#: of the string.
DEFAULT_BUCKET = "unidata-nexrad-level2"

#: The archive of record.  Still authoritative, still selectable with
#: ``--bucket``, and currently not granting anonymous access -- which is a
#: capability statement about the endpoint, not a judgement about which
#: bucket is the archive.
ARCHIVE_OF_RECORD_BUCKET = "noaa-nexrad-level2"

#: The mirror, under the name the earlier surface used for it.  It is the
#: default, so this is an alias and not a second choice.
MIRROR_BUCKET = DEFAULT_BUCKET

#: The real-time chunk feed's bucket, mirroring ``rw-nexrad``'s own
#: ``LIVE_DEFAULT_BUCKET``.  A different KEY SPACE from
#: :data:`DEFAULT_BUCKET` rather than another mirror of it -- one object per
#: LDM chunk under ``{SITE}/{VOLUME_ID}/`` instead of one Archive-II file
#: per finished volume -- which is why the live subcommands default to it
#: and ``list``/``fetch`` never do.
LIVE_DEFAULT_BUCKET = "unidata-nexrad-level2-chunks"

_PROBE_TIMEOUT_S = 20

#: ``CARGO_BUILD_HINT``: the one-liner that builds the decoder, from a
#: checkout root, spelled for the shell rule when it is read.
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)


def crate_dir() -> Path:
    """The vendored Rust workspace of a source checkout (may not exist)."""

    return Path(__file__).resolve().parent.parent.parent / "tools" / "rustwx"


def nexrad_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the front door, best first."""

    filename = executable_name(NEXRAD_NAME)
    candidates: list[Path] = []
    override = os.environ.get(NEXRAD_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def find_nexrad_bin() -> Path | None:
    """First existing candidate, or None.

    An environment override naming a missing file is a hard error.
    """

    override = os.environ.get(NEXRAD_ENV)
    for attempt in (0, 1):
        for candidate in nexrad_candidates():
            if candidate.is_file():
                return accept_resolved(candidate.resolve())
            if override and candidate == Path(override):
                raise FileNotFoundError(
                    f"{NEXRAD_ENV} names a missing file: {candidate}")
        # On demand since WOOF 1.1.6 (PyPI per-file limit): staged by the first
        # radar ingest from this release's bundle (never inside
        # bridges.inspection_only(), so doctor only reports).
        from woof.bridges import fetch_on_demand
        if attempt or override or not fetch_on_demand(NEXRAD_NAME):
            return None
    return None


def nexrad_remedy() -> str:
    """The remedy for a missing front door, true for THIS install."""

    return artifact_remedy(
        env_var=NEXRAD_ENV, filename=executable_name(NEXRAD_NAME),
        subject="the NEXRAD Level-II front door",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=rustwx_build_hint())


def probe_nexrad_bin(path: Path) -> tuple[bool, str]:
    """``--version`` then ``--abi``: is this binary the one we expect?"""

    try:
        version = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True,
            errors="replace", timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{path} did not run: {error}"
    if version.returncode != 0:
        return False, f"{path} --version exited {version.returncode}"
    transcript = (version.stdout or "").strip()
    try:
        abi = subprocess.run(
            [str(path), "--abi"], capture_output=True, text=True,
            errors="replace", timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{path} --abi did not run: {error}"
    if abi.returncode != 0 or (abi.stdout or "").strip() != NEXRAD_ABI_MARKER:
        return False, (f"{transcript} -- --abi does not match the record "
                       "contract this woof expects; rebuild it: "
                       f"{rustwx_build_hint()}")
    return True, f"{transcript} -- --abi matches the record contract"


def _run(command: list[str], *, what: str, schema: str) -> dict:
    """Run one ``rw_nexrad`` subcommand and parse its JSON record."""

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace")
    except OSError as error:
        raise RuntimeError(f"rw_nexrad failed to launch: {error}") from error
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip() and not line.startswith("usage:")
                and not line.startswith(" ")]
        detail = tail[0] if tail else f"exit {result.returncode}"
        raise RuntimeError(f"rw_nexrad {what}: {detail}")
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"rw_nexrad {what} did not print a JSON record: {error}"
        ) from error
    if record.get("schema") != schema:
        raise RuntimeError(
            f"rw_nexrad {what} printed schema {record.get('schema')!r}, "
            f"expected {schema!r}")
    return record


def _window(*, site: str, start: str, end: str, bucket: str | None,
            limit: int | None) -> list[str]:
    command = ["--site", site, "--start", start, "--end", end]
    if bucket is not None:
        command += ["--bucket", bucket]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def run_list(binary: Path, *, site: str, start: str, end: str,
             bucket: str | None = None, limit: int | None = None) -> dict:
    """List the volumes a (site, window) resolves to, moving no payload."""

    return _run([str(binary), "list",
                 *_window(site=site, start=start, end=end, bucket=bucket,
                          limit=limit)],
                what="list", schema=LIST_SCHEMA)


def run_fetch(binary: Path, *, site: str, start: str, end: str, out: Path,
              bucket: str | None = None, limit: int | None = None,
              cache_dir: Path | None = None,
              use_cache: bool = True) -> dict:
    """Download the volumes in the window into ``out``."""

    command = [str(binary), "fetch",
               *_window(site=site, start=start, end=end, bucket=bucket,
                        limit=limit),
               "--out", str(out)]
    if cache_dir is not None:
        command += ["--cache", str(cache_dir)]
    if not use_cache:
        command.append("--no-cache")
    return _run(command, what="fetch", schema=FETCH_SCHEMA)


def _live(*, site: str, bucket: str | None, volumes: int | None,
          volume_id: int | None, allow_partial: bool,
          min_chunks: int | None) -> list[str]:
    command = ["--site", site]
    if bucket is not None:
        command += ["--bucket", bucket]
    if volumes is not None:
        command += ["--volumes", str(volumes)]
    if volume_id is not None:
        command += ["--volume-id", str(volume_id)]
    if allow_partial:
        command.append("--allow-partial")
    if min_chunks is not None:
        command += ["--min-chunks", str(min_chunks)]
    return command


def run_live_list(binary: Path, *, site: str, bucket: str | None = None,
                  volumes: int | None = None, volume_id: int | None = None,
                  allow_partial: bool = False,
                  min_chunks: int | None = None) -> dict:
    """What the real-time chunk feed holds for a site right now.

    Moves no payload.  The record carries the bucket's own clock
    (``observed_at``) and a measured ``lag_seconds`` per volume, so a
    caller can compare feeds before choosing one.
    """

    return _run([str(binary), "live-list",
                 *_live(site=site, bucket=bucket, volumes=volumes,
                        volume_id=volume_id, allow_partial=allow_partial,
                        min_chunks=min_chunks)],
                what="live-list", schema=LIVE_LIST_SCHEMA)


def run_live_fetch(binary: Path, *, site: str, out: Path,
                   bucket: str | None = None, volumes: int | None = None,
                   volume_id: int | None = None, allow_partial: bool = False,
                   min_chunks: int | None = None,
                   cache_dir: Path | None = None,
                   use_cache: bool = True) -> dict:
    """Assemble the newest real-time chunks into an Archive-II volume.

    ``allow_partial`` is off by default here for the same reason it is off
    in the binary: a scan that is still being collected is a real product
    and not a silent one.  With it on, the published file is named
    ``..._P{NNN}`` -- a name the archive key parser refuses -- so a partial
    can never be picked up as a finished volume by anything downstream.
    """

    command = [str(binary), "live-fetch",
               *_live(site=site, bucket=bucket, volumes=volumes,
                      volume_id=volume_id, allow_partial=allow_partial,
                      min_chunks=min_chunks),
               "--out", str(out)]
    if cache_dir is not None:
        command += ["--cache", str(cache_dir)]
    if not use_cache:
        command.append("--no-cache")
    return _run(command, what="live-fetch", schema=LIVE_FETCH_SCHEMA)


def run_decode(binary: Path, *, volume: Path, out: Path,
               moments: tuple[str, ...] = ("REF", "VEL"),
               max_range_km: float | None = None,
               max_elevation_deg: float | None = None,
               site_latlon: tuple[float, float, float] | None = None,
               censor_flags: bool = False) -> dict:
    """Validate one volume and write a ``gpuwm-obs.radar-sweeps.v1`` pack.

    ``censor_flags`` asks for the ``v2`` pack instead: the same moment
    planes, plus a ``|u1`` plane per moment saying why each NaN gate is not
    a number.  Off by default, and with it off the pack is byte-identical to
    what this front door has always produced.
    """

    command = [str(binary), "decode", "--volume", str(volume),
               "--out", str(out), "--moments", ",".join(moments)]
    if censor_flags:
        command += ["--censor-flags"]
    if max_range_km is not None:
        command += ["--max-range-km", f"{max_range_km:g}"]
    if max_elevation_deg is not None:
        command += ["--max-elevation-deg", f"{max_elevation_deg:g}"]
    if site_latlon is not None:
        command += ["--site-latlon",
                    ",".join(f"{value:.10g}" for value in site_latlon)]
    return _run(command, what="decode", schema=DECODE_SCHEMA)


def run_verify(binary: Path, *, pack: Path) -> dict:
    """Re-prove a pack's header, schema and payload digest."""

    return _run([str(binary), "verify", "--volume", str(pack)],
                what="verify", schema=VERIFY_SCHEMA)


def run_sites(binary: Path, *, site: str | None = None) -> dict:
    """The vendored NEXRAD site table, or one row of it."""

    command = [str(binary), "sites"]
    if site is not None:
        command += ["--site", site]
    return _run(command, what="sites", schema=SITES_SCHEMA)
