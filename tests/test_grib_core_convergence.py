"""One GRIB decoder for the whole tree, and the pins that keep it one.

Until 2026-08-16 this repository carried TWO vendored `grib-core`
directories, both declaring ``name = "grib-core", version = "0.1.0"``, so
Cargo could not tell them apart and neither could a reader.  They had
diverged in one direction that mattered: the copy four of the five
consumer manifests pointed at -- the renderer and ingest path -- was the
copy WITHOUT ``missing_value_management``.

The concrete breakage each test below prevents is named on the test, and
they are all the same family of breakage: a decoder fix that lands in one
copy while the shipped path reads the other.  That is not hypothetical.
It is what let a complex-packed GRIB2 message (Template 5.2 or 5.3) that
declares in-band missing values decode the encoder's missing-value
markers as physical data on the renderer path for two release lines,
silently, with no error and no NaN.

These are text gates on purpose: they read manifests and a provenance
note, need no Rust toolchain and no card, and so they run on every
commit rather than at a cut.  What they cannot check is that the decoder
is correct -- that is the crate's own suite, and
``tools/battery/cargo_gates.txt`` is the leg that runs it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: The single converged crate.  Both workspaces reach this directory.
DECODER = REPOSITORY_ROOT / "tools" / "grib1_bridge" / "vendor" / "grib-core"

#: Its provenance record, carrying the upstream pins and the SHA-256 of
#: every local delta.
VENDOR_NOTE = DECODER.parent / "VENDOR.md"

#: Every consumer declaration, with the feature posture it had before the
#: convergence and must still have after it.  ``rw-wrfbatch`` and the
#: bridge build the decoder without the optional JPEG2000/PNG codecs;
#: the other three take defaults, which is what makes `openjp2` and `png`
#: part of the renderer's build closure.  A convergence that quietly
#: flipped any of these rows would either hard-require openjp2/png/cmake
#: where they were not required before, or drop a codec the renderer
#: decodes MRMS and Stage-IV with.
CONSUMERS = {
    "tools/rustwx/crates/rustwx-io/Cargo.toml": True,
    "tools/rustwx/crates/rustwx-products/Cargo.toml": True,
    "tools/rustwx/crates/rw-obs/Cargo.toml": True,
    "tools/rustwx/crates/rw-wrfbatch/Cargo.toml": False,
    "tools/grib1_bridge/Cargo.toml": False,
}

_DEPENDENCY = re.compile(
    r"^grib-core\s*=\s*\{\s*path\s*=\s*\"(?P<path>[^\"]+)\""
    r"(?P<rest>[^}]*)\}",
    re.MULTILINE,
)


def _manifest(relative: str) -> str:
    return (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")


def test_exactly_one_grib_core_crate_is_tracked() -> None:
    """A second copy is the defect, not a convenience.

    Breakage prevented: two crates named `grib-core` 0.1.0 that Cargo
    cannot distinguish, where a decoder fix lands in one and the shipped
    path compiles the other.
    """

    # TRACKED files, as the name says, not a filesystem walk: the MAIN
    # checkout hosts other checkouts inside ignored directories (every
    # agent worktrees, ~130 parked clones
    # under .worktrees), and each carries the pre-convergence layout of
    # whatever ref it has checked out.  A walk finds those and turns the
    # gate red in the one checkout the release battery actually runs in,
    # for an environmental reason.  A copy that is merely on disk but
    # untracked cannot ship and cannot be compiled unless a consumer
    # manifest points at it -- and the consumer tests below gate that.
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", "*grib-core/Cargo.toml"],
        cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    tracked = sorted(
        REPOSITORY_ROOT / entry
        for entry in listing.split("\0")
        if entry
    )
    assert tracked == [DECODER / "Cargo.toml"], (
        "expected exactly one vendored grib-core crate at "
        f"{DECODER.relative_to(REPOSITORY_ROOT)}, found "
        f"{[str(path.relative_to(REPOSITORY_ROOT)) for path in tracked]}"
    )


@pytest.mark.parametrize("relative", sorted(CONSUMERS))
def test_every_consumer_points_at_the_converged_crate(relative: str) -> None:
    """All five declarations resolve to the one directory.

    Breakage prevented: a consumer left pointing at a path that no longer
    exists (a build failure, which is loud) or -- far worse -- at a
    resurrected second copy (which is silent).
    """

    match = _DEPENDENCY.search(_manifest(relative))
    assert match, f"{relative} declares no grib-core path dependency"
    declared = (REPOSITORY_ROOT / relative).parent / match.group("path")
    assert declared.resolve() == DECODER.resolve(), (
        f"{relative} points at {declared} rather than the converged "
        f"{DECODER}"
    )


@pytest.mark.parametrize("relative", sorted(CONSUMERS))
def test_every_consumer_keeps_its_feature_posture(relative: str) -> None:
    """Convergence moved the crate, not the codecs each consumer builds.

    Breakage prevented: `default-features` flipped on during the move,
    which hard-requires `openjp2`/`png`/`cmake` in a build that did not
    need them; or flipped off, which drops the JPEG2000 and PNG codecs
    the renderer reads real products through.
    """

    match = _DEPENDENCY.search(_manifest(relative))
    assert match, f"{relative} declares no grib-core path dependency"
    takes_defaults = "default-features = false" not in match.group("rest")
    assert takes_defaults == CONSUMERS[relative], (
        f"{relative} default-features posture changed: expected "
        f"{'defaults on' if CONSUMERS[relative] else 'default-features = false'}"
    )


def test_vendor_sha_pins_match_the_vendored_files() -> None:
    """The provenance note describes the files that are actually there.

    Breakage prevented: a delta record that cannot be used to audit or
    reproduce the delta it claims to describe.  This is not a
    hypothetical either -- the `grib2/parser.rs` and `grib2/unpack.rs`
    pins in that note had already gone stale by `ed4c0ef69`, because
    nothing was checking them.
    """

    note = VENDOR_NOTE.read_text(encoding="utf-8")
    pinned = set(re.findall(r"`([0-9a-f]{64})`", note))
    assert pinned, "VENDOR.md records no per-file SHA-256 pins"

    delta_files = re.findall(r"`(src/[^`]+\.rs)`", note)
    assert delta_files, "VENDOR.md names no delta files"

    for relative in sorted(set(delta_files)):
        digest = hashlib.sha256((DECODER / relative).read_bytes()).hexdigest()
        assert digest in pinned, (
            f"{relative} hashes to {digest}, which VENDOR.md does not pin; "
            "refresh the note in the same commit that changes the file"
        )


def test_the_decoder_stays_inside_the_rust_1_75_floor() -> None:
    """The rental build floor is a property of the whole crate now.

    ``tools/grib1_bridge`` builds on a runtime pinned to Rust 1.75 with
    Cargo lock format 3, which is why six ``Option::is_none_or`` checks
    in `grib1/parser.rs` are spelled ``Option::map_or(true, ...)``.  That
    delta used to be one file's problem.  It is the whole crate's problem
    now that the renderer workspace -- which declares `rust-version =
    "1.85"` -- shares this source.

    Breakage prevented: an edit made under the 1.85 workspace's rules
    that the offline rental build cannot compile, discovered on the
    rental box rather than here.  The two spellings checked are the ones
    that actually bite: `is_none_or` stabilised in 1.82, let-chains in
    1.88.
    """

    offenders: list[str] = []
    for source in sorted(DECODER.rglob("*.rs")):
        text = source.read_text(encoding="utf-8")
        relative = source.relative_to(REPOSITORY_ROOT).as_posix()
        if "is_none_or" in text:
            offenders.append(f"{relative}: Option::is_none_or needs Rust 1.82")
        if re.search(r"&&\s*let\s", text):
            offenders.append(f"{relative}: let-chain needs Rust 1.88")
    assert not offenders, "; ".join(offenders)

    lockfile = (REPOSITORY_ROOT / "tools/grib1_bridge/Cargo.lock").read_text(
        encoding="utf-8")
    assert "version = 3" in lockfile and "version = 4" not in lockfile

    parser = (DECODER / "src/grib1/parser.rs").read_text(encoding="utf-8")
    assert parser.count("map_or(true, |end| end > data.len())") == 6


def test_the_missing_value_decode_is_default_on() -> None:
    """No feature flag stands between a reader and the fix.

    Breakage prevented: shipping the missing-value decode behind a Cargo
    feature, which under this project's law is a workaround rather than a
    fix -- a bare default build has to stop showing the defect.  The
    decoder's Cargo features are codec choices (JPEG2000, PNG, CCSDS) and
    nothing else, and `missing_value_mode` is reached from the plain
    Template 5.2/5.3 paths with no `cfg` on it.
    """

    manifest = (DECODER / "Cargo.toml").read_text(encoding="utf-8")
    features = re.search(r"\[features\](?P<body>.*)", manifest, re.DOTALL)
    assert features, "grib-core declares no [features] table"
    assert set(re.findall(r"^(\w+)\s*=", features.group("body"), re.MULTILINE)) == {
        "default", "jpeg2000", "png_codec", "ccsds",
    }

    unpack = (DECODER / "src/grib2/unpack.rs").read_text(encoding="utf-8")
    for line in unpack.splitlines():
        assert not (line.lstrip().startswith("#[cfg(") and "missing" in line), (
            f"missing-value decode sits behind a cfg: {line.strip()}")
    assert unpack.count("missing_value_mode(dr)?") >= 3, (
        "the two whole-field unpackers and the row-window unpacker each "
        "have to read Code Table 5.5 for themselves")
