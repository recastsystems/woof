"""A checkout's bytes are the committed bytes, for everything hashed.

git-for-Windows defaults to ``core.autocrlf=true``.  Without a
``.gitattributes`` saying otherwise, a Windows clone rewrites LF to CRLF
on checkout, so a file in a clone and the same file in a wheel are
different bytes.  That is invisible until something hashes them -- and
this product hashes a lot of them:

* ``_runtime_source_identity`` in both prepared-forecast runners binds
  the SHA-256 of the model sources it is about to run;
* the Thompson physics receipt binds its implementation files;
* the preparation proof binds an export-source set, and the forecast
  stage re-derives it and compares.

A mixed estate -- ``rw-wps`` from a pip venv, the runner from a fresh
clone -- therefore failed with ``proof export source hashes differ from
preparation``.  Nothing was corrupt; the two halves had different line
endings.

The remedy is ``* -text`` in ``.gitattributes``.  The gate is this file:
it asks git for the committed blob and compares it with what is on
disk.  On a Linux runner it is nearly tautological; on a Windows runner
it is the actual check, and Windows is where the failure lives.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)


def _is_checkout() -> bool:
    try:
        _git("rev-parse", "--show-toplevel")
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _hashed_paths() -> list[str]:
    """Every repository file the product hashes and then compares.

    Read from the runners themselves rather than re-listed here: a file
    added to one of those tuples has to be covered by this gate without
    anyone remembering to add it twice.
    """

    from woof import prepared_domain_tree_forecast as tree
    from woof import prepared_single_domain_forecast as single

    paths = {
        "woof/core/model.py",
        "woof/ingest/hrrr_physics.py",
        "woof/ingest/hrrr_surface.py",
        "woof/ingest/lateral_bc.py",
        "woof/ingest/prepared_cache.py",
        "woof/ingest/real.py",
        "woof/io/wrfout.py",
        "woof/native_wrf_contract.py",
        "woof/source_authorities.py",
        "woof/prepared_single_domain_forecast.py",
        "woof/prepared_domain_tree_forecast.py",
        "woof/core/nest.py",
        "woof/core/microphysics_transition.py",
        "woof/core/kernels/nest_microphysics.cu",
        "woof/physics_registry_v2.json",
        "woof/wrf_direct_v461_contract.json",
    }
    paths.update(single._THOMPSON_IMPLEMENTATION_FILES)
    # Named so a reader can see the two runners are the authority here.
    assert tree.RUNNER == "tools.prepared_domain_tree_forecast"
    return sorted(paths)


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_every_hashed_file_matches_its_committed_blob():
    """Disk bytes == object-database bytes, file by file."""

    drifted = []
    for relative in _hashed_paths():
        path = ROOT / relative
        if not path.is_file():
            continue  # an externalized asset; its own gate covers it
        try:
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{relative}"],
                cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue  # not committed yet on this branch
        if committed != path.read_bytes():
            drifted.append(relative)
    assert not drifted, (
        "these files differ from their committed blobs, which on Windows "
        "means core.autocrlf rewrote them on checkout and every hash the "
        "product takes of them now disagrees with the wheel's: "
        f"{drifted}")


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_every_hashed_blob_is_lf_only():
    """The committed blobs themselves carry no CR, absolutely.

    The disk-vs-blob comparison above is necessary but blind to one
    arrangement: a file committed WITH CRLF checks out as CRLF and the
    two sides agree byte for byte.  That is exactly what a Windows
    worktree with ``core.autocrlf=true`` produces when a lane edits a
    hashed file -- the flip enters the object database and every hash
    the product takes of it disagrees with a Linux clone's, while the
    gate above stays green on the machine that caused it.  So the blob
    is asked directly: zero CR bytes, no allowlist, no tolerance.
    """

    crlf = []
    for relative in _hashed_paths():
        try:
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{relative}"],
                cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue  # not committed yet on this branch
        if b"\r" in committed:
            crlf.append(relative)
    assert not crlf, (
        "these committed blobs contain CR bytes; a CRLF blob hashes "
        "differently on every platform that checks out LF, so the "
        "runtime source identity splits by clone: "
        f"{crlf}")


# ---------------------------------------------------------------------------
# The hashed set is not the whole promise
# ---------------------------------------------------------------------------
#
# The two gates above cover `_hashed_paths()` -- about twenty files -- and
# `* -text` promises the whole repository.  The gap between those two is
# not hypothetical: between v2.3.3 and v2.4.1 the object database gained
# CRLF in thirty-four files, sixteen of them existing files flipped whole
# (`woof/cli.py`, `woof/doctor.py`, `woof/bridge_assets.py`,
# `pyproject.toml`, `CHANGELOG.md`, eight test modules and more) and
# eighteen of them newly authored that way.  Every one of the three tests
# above stayed green, because not one of those files is hashed.
#
# The damage was not a hash split, it was the merge: folding v2.4.1 onto
# the 2.5.0 line produced line-for-line conflicts across whole files --
# `woof/doctor.py` alone is 3,859 lines -- with the real semantic
# conflicts invisible underneath them.  So the gate widens to the surface
# this repository AUTHORS, and the recorded debt below is what it found
# there when it was written.

#: Trees whose bytes are deliberately preserved as they were received, so
#: their line endings are not this repository's to normalize.
#: `woof/data/**`, `tools/*/vendor/**` and `patches/**` are the same
#: three `.gitattributes` names them individually for.
_BYTE_PRESERVED_PREFIXES = (
    "woof/data/",
    # The GF, New Tiedtke, urban, UH and Noah mosaic oracle fixtures, moved
    # out of woof/data/ in 2.8.6 with their bytes unchanged: captured WRF
    # output whose raw .bin words and receipts contain CR bytes as data.
    "tests/data/oracles/",
    "tools/rustwx/vendor/",
    "tools/grib1_bridge/vendor/",
    "patches/",
)

# Only this exact captured receipt and digest retain original line endings.
_BYTE_PRESERVED_FILES = {
    "tests/fixtures/tolerance_parent_byte_controls/original-recipe-lineage.json":
        "de1dc8d48077bacfa9cf5b591ef0a8b89ea33d4997519295ed27f9ad0b6c53c4",
}


def test_captured_lineage_receipt_retains_its_original_digest():
    for relative, expected in _BYTE_PRESERVED_FILES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        if _is_checkout():
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{relative}"], cwd=ROOT)
            assert hashlib.sha256(committed).hexdigest() == expected

#: The rustwx crates this repository authors, identified the same way
#: `tests/test_bridge_source_rev_stamp.py` identifies them: a crate this
#: repository stamps with its own source revision is a crate it wrote.
#: Everything else under `tools/rustwx/crates/` is the verbatim subset of
#: the upstream rusty-weather workspace that `tools/rustwx/Cargo.toml`
#: documents, vendored at an exact working-tree state.  `rw-host-memory`
#: stamps no revision but is authored here: the host-memory walk moved out
#: of rusty-weather into its own crate.  So is `rw-libm`, the platform-
#: independent libm the WPS orographic projection calls.
_AUTHORED_RUSTWX_CRATES = (
    "rw-fetch", "rw-wrfbatch", "rw-nexrad", "rw-odim", "rw-obs",
    "rw-goes", "rw-netcdf", "rw-mpas", "rw-host-memory",
    "rw-isobaric", "rw-mlexport", "rw-libm", "rw-post",
)

#: The 62 files that already carried CR when this gate was widened,
#: recorded so the gate can be a ratchet instead of a 62-file cleanup
#: nobody asked for in a merge, across an authored surface of 1,827
#: files.  This list may only shrink.  Nothing here is blessed:
#: it is debt with a name, and `test_the_crlf_debt_does_not_rot` fails
#: the moment an entry is fixed or deleted without being struck off.
_CRLF_DEBT = frozenset({
    # the repository root: .gitignore left this list in the 3.0 fold (the
    # public-aerosol merge 00c2d5939cd wrote it back LF); CHANGELOG.md and
    # pyproject.toml were flipped by the 2.4 line and are LF again.
    # gpuwm/ -- the wheel's own sources.  woof/core/kernels/shinhong.cu
    # left this list at A146: its constant-divisor rewrite (a98f2482e) wrote
    # the file back LF, and restoring the CR bytes is a commit the line-ending
    # hook refuses, so the file stays normalized.
    "woof/core/rrtm_taumol.py",
    # woof/da/letkf.py left this list in the DA speed integration: the
    # obs-ops lane wrote it back LF (e13abf285) and the line-ending hook
    # refuses restoring the CR bytes, so the file stays normalized.
    "woof/da/obs_goes.py",
    # woof/da/obs_radar.py left this list when the DA line merged into
    # 2.8.7: the radar reflectivity rebuild (098029d01, DA lane 2) wrote it
    # back LF together with its content change, and the line-ending hook
    # refuses restoring the CR bytes, so the file stays normalized.
    "woof/obs/goes_cwp.py",
    "woof/obs/goes_grid.py",
    "woof/obs/goes_pack.py",
    "woof/verify/cases/nest_spawn_ideal.py",
    "woof/verify/feedback_reference.py",
    "woof/verify/obs/battery.py",
    "woof/verify/obs/fss.py",
    "woof/verify/obs/promotion.py",
    "woof/verify/obs/registration.py",
    "woof/verify/obs/stubs.py",
    # tests/
    "tests/data/wrfout_global_attribute_set_v1.json",
    "tests/goes_pack_fixtures.py",
    "tests/test_goes_cwp.py",
    "tests/test_grayzone_nest_config.py",
    "tests/test_inflow_fetch_meter.py",
    "tests/test_les_tornado_attempt1_config.py",
    "tests/test_obs_battery.py",
    # tests/test_obs_battery_ingest.py left this list in 2.9.2: the radar
    # feeds lane (47c97c0c) wrote it back LF with its one-line change, and
    # the line-ending hook refuses restoring the CR bytes, so the file stays
    # normalized.
    "tests/test_obs_controls.py",
    "tests/test_obs_model_source.py",
    "tests/test_obs_promotion.py",
    "tests/test_render_basemap_delivery.py",
    "tests/test_spawn_runner.py",
    "tests/test_tke_km2.py",
    # tools/
    "tools/emit_les_route_inputs.py",
    "tools/inflow_fetch_meter.py",
    "tools/obs_battery_score.py",
    "tools/obs_fetch_asos.py",
    # tools/obs_fetch_mrms.py left this list when the nowcast lane rewrote it
    # onto woof.obs.mrms_fetch: the rewrite wrote LF, so the debt is settled
    # rather than waived.
    "tools/obs_fetch_stage4.py",
    "tools/obs_goes_grid_build.py",
    "tools/obs_radar_grid_build.py",
    "tools/obs_radar_grid_from_pack.py",
    "tools/rustwx/crates/rw-fetch/Cargo.toml",
    "tools/rustwx/crates/rw-wrfbatch/src/grib_import.rs",
    "tools/rustwx/crates/rw-wrfbatch/src/wrf_volumes.rs",
    # docs/ -- receipts are transcripts and sit outside the surface
    "docs/les/ATTEMPT1-EXPECTATIONS.md",
    "docs/les/attempt1/eta49-shipped-control.json",
    "docs/les/attempt1/eta72-ladder.json",
    "docs/les/attempt2/eta72-raisedlid.json",
    "docs/obs-goes-cwp-bridge-design.md",
    "docs/public/LES.md",
    # configs/
    "configs/frozen/les_nest_250m_km3.toml",
    "configs/les_nest_250m_km3.toml",
    "configs/moving_nest_20110427_move_proof.toml",
})


def _is_authored(relative: str) -> bool:
    """Is this path one this repository wrote, rather than received?"""

    if relative in _BYTE_PRESERVED_FILES:
        path = ROOT / relative
        if (path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()
                == _BYTE_PRESERVED_FILES[relative]):
            return False
    # Raw IEEE float fixtures have no text line endings. Small captures can
    # contain a CR byte without any NUL, so git's content heuristic is not
    # sufficient to distinguish them from authored source text.
    if relative.endswith(".f32"):
        return False
    if any(relative.startswith(p) for p in _BYTE_PRESERVED_PREFIXES):
        return False
    if relative.startswith("docs/"):
        # Receipts are captured transcripts of other tools' output.
        return "/receipts/" not in relative
    if relative.startswith("tools/rustwx/crates/"):
        crate = relative.split("/")[3]
        return crate in _AUTHORED_RUSTWX_CRATES
    if relative.startswith("tools/"):
        return relative.endswith(".py")
    if relative.startswith(("woof/", "tests/", "configs/")):
        return True
    return "/" not in relative  # the release mechanics at the repo root


#: Handed to git as pathspecs so it never opens the vendored trees at
#: all.  `_is_authored` is still the authority -- this only keeps the
#: scan off the ~2.5 GB of vendored crates.
_SCAN_EXCLUDES = tuple(f":!{prefix}" for prefix in _BYTE_PRESERVED_PREFIXES)


def _authored_paths() -> set[str]:
    """Every authored path committed at HEAD."""

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "HEAD"],
        cwd=ROOT, capture_output=True, check=True).stdout
    found = set()
    for record in listing.split(b"\x00"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        _mode, kind, _sha = meta.split(b" ")
        if kind != b"blob":
            continue
        relative = raw_path.decode("utf-8", "surrogateescape")
        if _is_authored(relative):
            found.add(relative)
    return found


def _cr_bearing_authored() -> set[str]:
    """Authored paths whose COMMITTED blob contains a CR byte.

    ``git grep`` rather than reading every blob here: it is one process
    against the object database instead of a thousand, it already skips
    what it considers binary (``-I``), and the alternative -- feeding
    ``git cat-file --batch`` a thousand object ids down a pipe while
    nothing drains its output -- deadlocks on a 64 KB pipe buffer, which
    is a slow way to learn that a gate is a test too.
    """

    found = subprocess.run(
        ["git", "grep", "-l", "-z", "-I", "-F", "-e", "\r",
         "HEAD", "--", *_SCAN_EXCLUDES],
        cwd=ROOT, capture_output=True)
    if found.returncode not in (0, 1):  # 1 is git grep for "no matches"
        raise AssertionError(
            f"git grep failed: {found.stderr.decode('utf-8', 'replace')}")
    reported = found.stdout.decode("utf-8", "surrogateescape")
    # Each record is `HEAD:<path>`; the revision is fixed, so split once.
    paths = {record.split(":", 1)[1]
             for record in reported.split("\x00") if record}
    return {relative for relative in paths if _is_authored(relative)}


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_no_authored_file_gains_a_carriage_return():
    """The `* -text` promise, held across everything this repo writes.

    A per-path allowlist covering only the hashed files is what let
    thirty-four files through between two releases.  This asks the whole
    authored surface and names the difference against the recorded debt,
    so the next flip is one file in a failure message rather than twenty
    thousand lines in a merge.
    """

    fresh = sorted(_cr_bearing_authored() - _CRLF_DEBT)
    assert not fresh, (
        "these committed blobs gained CR bytes and are not recorded debt. "
        "A Windows worktree with core.autocrlf=true does this on its own, "
        "and the flip is invisible until a merge or a hash disagrees. "
        "Normalize them to LF -- the change must be CR-only -- rather "
        f"than adding them below: {fresh}")


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_the_crlf_debt_does_not_rot():
    """The debt list may only shrink, and only accurately.

    An entry that has been fixed or deleted has to be struck off in the
    same commit, otherwise the list slowly becomes a description of a
    repository that no longer exists and stops being evidence of
    anything.
    """

    present = _authored_paths()
    carrying = _cr_bearing_authored()
    settled = sorted(entry for entry in _CRLF_DEBT
                     if entry in present and entry not in carrying)
    departed = sorted(entry for entry in _CRLF_DEBT if entry not in present)
    assert not settled, (
        "these files are LF now, so strike them off _CRLF_DEBT in the "
        f"commit that fixed them: {settled}")
    assert not departed, (
        "these _CRLF_DEBT entries are no longer authored files in this "
        f"tree; remove them: {departed}")


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_the_repository_declares_no_conversion_at_all():
    """``* -text`` is present, so a new file is covered by default.

    A per-path allowlist is the arrangement that produced the bug: it
    covered the JSON and the shell scripts and not ``woof/*.py``, which
    is what the runners hash.  The catch-all is the fix; this holds it.
    """

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    lines = [line.strip() for line in attributes.splitlines()]
    assert "* -text" in lines

    # And git agrees, for a file nothing names individually.
    reported = _git("check-attr", "text", "--", "woof/core/model.py")
    assert reported.strip().endswith(": text: unset"), reported


# ---------------------------------------------------------------------------
# The blob is not the whole promise either
# ---------------------------------------------------------------------------
#
# Everything above reads the OBJECT DATABASE, and `.gitattributes` runs two
# regimes that fail differently under it.  `* -text` covers most of the
# tree: no conversion in either direction, so a CRLF write lands in the
# blob verbatim and `test_no_authored_file_gains_a_carriage_return` sees
# it.  Twenty explicit `text eol=lf` rules are the other regime -- git
# NORMALIZES those on the way in, so the blob stays clean while the
# working tree stays CRLF, and every test above is structurally blind to
# the whole class.  Measured: across all 127 commits that touched
# `woof/physics_registry_v2.json`, not one blob has ever carried a CR
# byte, so no amount of blob-reading could ever have caught it.
#
# The damage is a digest split rather than a merge.  The 52
# `woof/authorities/*.json` are in this class, and `packaged_authorities`
# in `woof/source_authorities.py` resolves each one, takes
# `hashlib.sha256(path.read_bytes())` OFF DISK, and raises `packaged
# <profile> <role> authority hash differs` when it does not match the
# pinned contract -- as does `packaged_contributing_mappings` beside it.
# A CRLF working-tree copy is a different digest, so it raises, on a
# machine whose blobs are all correct.  The prepared-forecast proof
# re-derives its export-source digests from disk in the same way, which
# is the `proof export source hashes differ from preparation` that
# `.gitattributes` was written to prevent -- while every blob-reading
# gate above stays green on the machine that caused it.
# `woof/physics_registry_v2.json` surfaced exactly this way, by accident,
# as a disk-versus-blob mismatch inside an unrelated byte-reproduction
# test.
#
# So the gate below reads the WORKING TREE, for exactly the normalizing
# set, and derives that set at runtime from `git check-attr` rather than
# restating the twenty patterns -- a copied list drifts from
# `.gitattributes` the moment somebody adds the twenty-first.


def _normalizing_paths() -> set[str]:
    """Tracked paths git normalizes on the way into the object database.

    `text`, not `eol`, because the LAST matching rule in `.gitattributes`
    wins: 283 tracked paths carry `eol=lf` and only 260 of them also have
    `text` set -- the other 23 are `*.sh` under `tools/rustwx/crates/**`,
    which a later `-text` takes back.  `text` set is what actually makes
    git convert, so it is the attribute asked; keying on `eol` would put
    those 23 on the wrong side.
    """

    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            capture_output=True, check=True).stdout
    paths = [entry for entry
             in listed.decode("utf-8", "surrogateescape").split("\x00")
             if entry]
    asked = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "text"], cwd=ROOT,
        input="\x00".join(paths).encode("utf-8", "surrogateescape"),
        capture_output=True, check=True).stdout
    fields = asked.decode("utf-8", "surrogateescape").split("\x00")
    # `-z` output is a flat NUL-separated stream of (path, attr, value).
    return {fields[i] for i in range(0, len(fields) - 2, 3)
            if fields[i + 2] == "set"}


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_no_normalized_file_carries_a_carriage_return_on_disk():
    """For the `text eol=lf` set, the working tree is the only witness.

    NO ALLOWLIST, and that is deliberate rather than strict: `_CRLF_DEBT`
    above records "this blob carries CR", which is a different fact and
    cannot be true here at all -- git strips the CR before the blob
    exists.  There is no such thing as recorded debt in this class, so a
    hit is always fresh and the remedy is always to normalize the file.
    Measured when this was written: 260 tracked paths normalize, 60 of
    them are authored, and none of the 260 carried a CR on disk, so this
    is a ratchet from its first commit and not a cleanup.
    """

    carrying = sorted(
        relative for relative in _normalizing_paths()
        if _is_authored(relative)
        and (ROOT / relative).is_file()
        and b"\r" in (ROOT / relative).read_bytes())
    assert not carrying, (
        "these files carry CR in the WORKING TREE while `text eol=lf` "
        "keeps their committed blobs clean, so no blob-reading gate can "
        "see them.  The product hashes several of them from disk -- "
        "`packaged_authorities` sha256s each woof/authorities/*.json "
        "off disk against its pinned contract, and the prepared-forecast "
        "proof re-derives its export-source digests the same way -- so "
        "this clone raises `authority hash differs` or `proof export "
        "source hashes differ from preparation` against a wheel built "
        "from the same commit.  Normalize them to LF; there is no debt "
        f"list for this class: {carrying}")

    # The 200 normalizing paths outside `_is_authored` -- the vendored
    # `.sh` under `patches/`, `woof/data/`, the FTZ receipt trees -- are
    # deliberately out of scope, so both halves of this gate keep one
    # definition of "ours" instead of two that drift.


# ---------------------------------------------------------------------------
# And a gate nobody runs is not a gate
# ---------------------------------------------------------------------------


def _hook_module():
    """`tools/git_hooks/pre_commit_line_endings.py`, loaded by path."""

    import importlib.util

    source = ROOT / "tools" / "git_hooks" / "pre_commit_line_endings.py"
    spec = importlib.util.spec_from_file_location("_hook_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_the_pre_commit_hook_is_armed():
    """The write-side hook is installed in this clone, right now.

    THE BREAKAGE THIS NAMES is the one that already happened twice.  The
    hook was reachable only behind an `--install` flag documented inside
    its own docstring; nothing in the repository ran it, so
    `.git/hooks/pre-commit` did not exist at all, and two commits walked
    straight through the check written to stop them:
    `tests/test_offline_child.py` rewritten to CRLF across all 673 lines
    with a real 79-line addition buried inside the whole-file diff
    (repaired by 39ef138c5), and
    `tools/rustwx/crates/rw-mpas/src/bin/fp32_floor_probe.rs` born with
    525 carriage returns (repaired by abb5ff270).  The release gates in
    this file catch that class only at release time, by which point the
    bytes are landed history and the cost is a whole-file merge conflict
    rather than one sed.

    `conftest.py` at the repository root arms the hook on every pytest
    session, so this passes without anyone doing anything.  It is a
    separate question from the filesystem rather than the same call
    twice: if arming failed -- a read-only `.git`, or a `pre-commit`
    belonging to something else, which is never overwritten -- this is
    where that becomes visible.  A hook that is quietly not there is the
    defect being fixed, so its absence must fail rather than pass.
    """

    hook = _hook_module()
    installed = hook.hooks_dir() / "pre-commit"
    assert installed.is_file(), (
        "no pre-commit hook in this clone, so a patch script can flip a "
        "whole file to CRLF and commit it unopposed -- which is exactly "
        f"how 39ef138c5 and abb5ff270 came to be needed.  Arm it: {installed}"
        " is written by conftest.py at the repository root, or by "
        "`python tools/git_hooks/pre_commit_line_endings.py --install`")
    assert installed.read_bytes() == hook.hook_shim().encode("utf-8"), (
        f"{installed} exists but is not the shim this repository "
        "installs, so what runs at commit time is not this check.  If it "
        "is another tool's hook, chain this one from it; if it is a stale "
        "copy, re-run `python tools/git_hooks/pre_commit_line_endings.py "
        "--install`")


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_the_hook_and_the_gate_agree_on_which_paths_git_normalizes():
    """One answer to "does git convert this path", not two.

    The hook decides per staged path whether to ask the INDEX or the
    WORKING TREE, and it is the working-tree branch that sees the
    `text eol=lf` class at all.  If that classification drifts from this
    file's -- a `-z` parse that silently returns nothing, an `eol=lf`
    read where `text` is what actually converts -- the branch stops
    firing and the hook goes quietly back to being blind to the class,
    with every test still green.  That is the failure mode this whole
    change exists to remove, so the two are compared directly rather
    than each being trusted on its own.
    """

    hook = _hook_module()
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    paths = [entry for entry
             in tracked.decode("utf-8", "surrogateescape").split("\x00")
             if entry]

    assert hook.normalizing(paths) == _normalizing_paths()

    # And the set is not empty, because two functions that both return
    # nothing also "agree".  `woof/physics_registry_v2.json` is the file
    # that surfaced this class, by accident, as a disk-versus-blob
    # mismatch inside an unrelated byte-reproduction test.
    assert "woof/physics_registry_v2.json" in _normalizing_paths()
    assert "woof/core/model.py" not in _normalizing_paths()


def test_the_hook_and_the_gate_agree_on_what_binary_means():
    """A binary blob is outside the promise, on BOTH sides of it.

    The release scan (`_cr_bearing_authored`) runs `git grep -I`, which
    skips blobs git considers binary, so a binary path can never appear
    in its answer -- and therefore can never be recorded in `_CRLF_DEBT`
    without `test_the_crlf_debt_does_not_rot` striking it as settled on
    the next run.  The hook must draw the same line or it refuses
    commits whose only lawful remedy it itself names and this file
    forbids: rw-netcdf's checked-in NetCDF-4 fixtures are HDF5, whose
    signature contains `\\r\\n` by specification, and the hook refused
    them as fresh carriage returns.  git's binary test is a NUL in the
    leading window; the hook's predicate is pinned to it here, against
    the HDF5 signature that surfaced the drift.
    """

    hook = _hook_module()
    hdf5_signature = b"\x89HDF\r\n\x1a\n"
    assert hook._blob_is_binary(hdf5_signature + b"\x00" * 8)
    assert not hook._blob_is_binary(b"plain text\r\nwith crlf\r\n"), (
        "a text blob must stay inside the ratchet; only content git "
        "itself would call binary may pass")


def test_raw_float_capture_without_nul_does_not_hide_a_text_crlf_flip(monkeypatch):
    from types import SimpleNamespace
    hook = _hook_module()
    fixture = "tools/rustwx/crates/rw-wrfbatch/src/fixtures/visibility.f32"
    source = "tools/rustwx/crates/rw-wrfbatch/src/visibility.rs"
    # One valid IEEE float word has a CR byte and no NUL. This is numeric
    # fixture content, not a Windows text-mode rewrite.
    word = b"\x0d\x1a\x33\x42"
    assert not hook._blob_is_binary(word)
    monkeypatch.setattr(hook,"_authority",lambda: SimpleNamespace(_is_authored=_is_authored))
    monkeypatch.setattr(hook,"_staged_paths",lambda: [fixture,source])
    monkeypatch.setattr(hook,"normalizing",lambda paths: set())
    monkeypatch.setattr(hook,"_blob",lambda rev,path: None if rev=="HEAD" else word if path==fixture else b"source\r\n")
    assert hook.offenders() == [(source,1,"new file")]


@pytest.mark.skipif(not _is_checkout(),
                    reason="line-ending stability is a checkout property")
def test_arming_is_idempotent_and_never_clobbers_a_hook_it_did_not_write(
        tmp_path, monkeypatch):
    """Default-on has to be safe to run on every pytest session.

    `conftest.py` calls `ensure_installed()` unconditionally, in every
    lane, in a clone with 192 worktrees (censused 2026-08-29) that all
    share one hooks directory.  Two things have to hold for that to be
    a fix rather than a new hazard.  It must be idempotent, or a test
    run rewrites the hook every time and any other tool watching that
    file sees churn.
    And it must refuse a `pre-commit` it did not write: silently
    replacing somebody else's hook is a worse failure than the CRLF
    flips being prevented, and it would be invisible until whatever that
    hook did stopped happening.

    The refusal is reported rather than swallowed -- `conftest.py` turns
    a False into a RuntimeWarning and
    `test_the_pre_commit_hook_is_armed` turns it into a red test -- so a
    clone where arming could not happen says so, instead of quietly not
    having a hook, which is the original defect.
    """

    hook = _hook_module()
    monkeypatch.setattr(hook, "hooks_dir", lambda: tmp_path)
    shim = hook.hook_shim().encode("utf-8")

    armed, _story = hook.ensure_installed()
    assert armed
    assert (tmp_path / "pre-commit").read_bytes() == shim

    armed, story = hook.ensure_installed()
    assert armed and "already armed" in story

    foreign = b"#!/bin/sh\n# another tool's pre-commit hook\nexit 0\n"
    (tmp_path / "pre-commit").write_bytes(foreign)
    armed, story = hook.ensure_installed()
    assert not armed, story
    assert (tmp_path / "pre-commit").read_bytes() == foreign


def test_the_installed_shim_is_a_no_op_where_the_script_is_absent():
    """The hooks directory is per CLONE, and most worktrees predate this.

    `git rev-parse --git-path hooks` from a linked worktree answers with
    the MAIN checkout's hooks, shared by every worktree cut from it.
    Censused on 2026-08-29: 192 worktrees share this clone's hooks and
    only 72 contain `tools/git_hooks/pre_commit_line_endings.py` -- the
    rest sit on branches that predate it.  A shim that simply `exec`s the
    script exits 2 there with "can't open file" and REFUSES EVERY COMMIT
    in 120 working trees, saying nothing about line endings.  Arming by
    default is only safe because of this one line, so it is pinned here.
    """

    lines = [line.strip() for line in _hook_module().hook_shim().splitlines()]
    assert '[ -f "$script" ] || exit 0' in lines

    # And the interpreter is made to RUN before it is trusted.  `command
    # -v python3` succeeds on this machine and resolves to the Windows App
    # Execution Alias, which prints "Python was not found" and exits 49 --
    # a shim that only checked PATH refused a real commit on its first
    # try, which is a different way of arming nothing.
    assert 'if "$py" -c "" >/dev/null 2>&1; then' in lines

def test_the_arming_conftest_only_claims_hooks_it_owns(monkeypatch):
    """An unpacked sdist inside a stranger's repo must not arm anything.

    ``rev-parse --git-dir`` answers "yes" from ANY directory under ANY
    repository, so the old check would have written our pre-commit hook
    into a git directory we do not own the day someone ran pytest over an
    sdist unpacked inside their own project.  The ownership tie is the
    toplevel comparison: arm only when the working tree's root IS the
    directory holding the root conftest.
    """
    import importlib.util
    import pathlib as _pathlib
    import subprocess as _subprocess

    root = _pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gpuwm_root_conftest_probe", root / "conftest.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    # In this checkout the toplevel IS the conftest's directory.
    assert probe._is_checkout() is True

    # A stranger's repository: rev-parse succeeds, toplevel is elsewhere.
    real_run = _subprocess.run

    def foreign_toplevel(cmd, **kwargs):
        if "--show-toplevel" in cmd:
            class Done:
                returncode = 0
                stdout = str(root.parent).encode()
                stderr = b""
            return Done()
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(probe.subprocess, "run", foreign_toplevel)
    assert probe._is_checkout() is False
