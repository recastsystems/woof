"""``woof fetch-tables``: stage externalized physics table assets.

Most packaged lookup tables ship inside the wheel and the repository.
The exceptions are the *externalized* assets named here: files whose
size exceeds distribution-channel limits, published as versioned GitHub
release assets and staged into the resolved table root on demand.
Currently that is the two largest classic Thompson coefficient tables:

- ``freezeH2O.dat`` (243 MiB): over GitHub's 100 MiB blob limit, so it
  is absent from both the repository and the wheel/sdist -- every
  install fetches it once.
- ``qr_acr_qg_V4.dat`` (71 MiB): fits in the repository (git-clone
  installs already have it; fetch-tables verifies and moves on) but
  would push the wheel and sdist over PyPI's 100 MB per-file cap, so
  wheel installs fetch it.

The staging contract is the model's own: every byte that lands in the
table root is verified against the exact size and SHA-256 pins in
:data:`woof.core.thompson_contract.CLASSIC_TABLE_ASSETS` -- the same
pins every ``mp_physics=8`` run enforces at load -- *before* the file is
atomically moved into place.  A mismatched download is deleted and
refused, never installed. A damaged asset in the self-chosen staging
root is replaced atomically from a verified packaged copy; an explicit
mirror with wrong bytes is refused. The command is
idempotent: with all assets present and byte-valid it verifies and
exits 0 without touching the network.

``--from DIR`` stages from a local directory (offline/air-gapped
installs) under the same verification.  ``WOOF_TABLE_ASSET_URL_BASE``
overrides the download base URL (mirrors); the asset filename is
appended to it either way.

``--wif`` adds ONE more externalized asset to the same transaction: the
225 MB ``QNWFA_QNIFA_SIGMA_MONTHLY.dat`` global monthly aerosol
climatology the mp=28 WIF ingest reads
(:mod:`woof.ingest.wif_dataset` holds its pin and its acquisition
route).  It is opt-in rather than default because it is an INPUT
DATASET, not a coefficient table: no scheme selection makes it
mandatory, only ``aer_init_opt=1`` with ``wif_input_opt=1`` does, and a
default install must not be asked for a quarter-gigabyte it will never
open.  It stages into ``~/.woof/wif`` -- its own root, because a table
root is validated as a complete set and a WPS intermediate file inside
one would make "complete" unanswerable -- under the identical
verify-then-atomically-install contract, since the reason both files
are externalized is one reason: the release checklist's size gates.

WHERE the bytes land moved once, deliberately.  This command used to
stage into whatever ``thompson_table_root()`` resolved to, which on a
wheel install is *inside site-packages* -- so the 315 MiB a user had
just paid for was deleted by the next ``pip install --upgrade`` or venv
rebuild, silently, and the failure surfaced as a bare
``FileNotFoundError`` traceback in the middle of a forecast that had
already paid for fetch and preprocessing.  Staging now targets
``~/.woof/tables/thompson`` (:func:`woof.physics_compat
.user_thompson_table_root`), beside ``~/.woof/bridges``, and the
staged root is made *complete*: the assets that do ship in the package
are copied in beside the downloaded ones, because a root is validated
as a whole set and a half-staged directory would resolve to nothing.
An install whose packaged root is already complete -- every git clone,
and every wheel that staged into site-packages before this change --
is left exactly where it is.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tomllib
import uuid

from woof import bridges
from woof import fetch_guard
from woof.progress import ByteCounter
from woof.core.thompson_contract import (
    CLASSIC_TABLE_ASSETS,
    TableAsset,
    validate_table_assets,
)
from woof.geog_assets import _rw_asset, GeogFetchError
from woof.timeline.ledger import span

TABLE_SOURCE_PATH = Path(__file__).parent / "data" / "table_sources.v1.toml"


def table_source_rows(asset: TableAsset) -> tuple[dict, ...]:
    """Mirror choices are metadata. Every choice retains the canonical raw pin."""
    with TABLE_SOURCE_PATH.open("rb") as source:
        raw = tomllib.load(source)
    if raw.get("schema") != "gpuwm-table-sources-v1":
        raise TableAssetError("table source schema is unknown; source versions cannot be bound to the pinned coefficients")
    return tuple(row for row in raw["source"] if row["filename"] == asset.filename
                 and row["sha256"] == asset.sha256 and row["bytes"] == asset.bytes)


def fetch_asset_from_sources(root: Path, asset: TableAsset) -> Path:
    override = os.environ.get(ASSET_URL_BASE_ENV, "").strip()
    if not override:
        override = os.environ.get("WOOF_BRIDGE_ASSET_URL_BASE", "").strip() if asset.filename not in EXTERNALIZED_TABLE_FILENAMES else ""
    if override:
        return _fetch_asset_from_url(root, asset, override.rstrip("/") + "/" + asset.filename)
    rows = table_source_rows(asset)
    if not rows:
        raise TableAssetError(f"{asset.filename}: no source row; fetching an unspecified table version would change lookup coefficients")
    errors = []
    for row in rows:
        try:
            print(f"woof fetch-tables: {asset.filename}: trying pinned source {row['url']}",
                  file=sys.stderr)
            url = row["url"]
            if url.startswith("package://"):
                import importlib
                package, _, relative = url[len("package://"):].partition("/")
                if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise TableAssetError(f"{asset.filename}: invalid packaged source path; it would read unrelated bytes outside the data package")
                try:
                    source = importlib.import_module(package).data_root() / relative
                except (ImportError, AttributeError) as error:
                    raise TableAssetError(f"{asset.filename}: {package} cannot locate its pinned data: {error}") from error
                url = Path(source).resolve().as_uri()
            return _fetch_asset_from_url(root, asset, url, compression=row.get("compression"))
        except TableAssetError as error:
            # Corruption is a refusal, never a reason to silently select another mirror.
            if "pin refused" in str(error) or "pinned object refused" in str(error):
                raise
            errors.append(str(error))
    raise TableAssetError(f"{asset.filename}: no declared source delivered the pinned bytes: " + "; ".join(errors))

#: Externalized asset filenames: published as release assets and
#: excluded from the wheel/sdist (freezeH2O.dat is additionally absent
#: from the repository -- GitHub rejects blobs over 100 MiB).
#: Other classic files ship in the package. Their pinned source rows also
#: support empty explicit roots; an undeclared gap still names an incomplete
#: install rather than fetching unspecified lookup coefficients.
EXTERNALIZED_TABLE_FILENAMES = frozenset({
    "freezeH2O.dat",
    "qr_acr_qg_V4.dat",
})

#: Release the pinned assets were published with.  The pins themselves
#: live in thompson_contract; this only says where the bytes are hosted.
RELEASE_ASSET_BASE_URL = (
    "https://github.com/recastsystems/woof/releases/download/v1.0.0")

#: Fixed data release for the unchanged monthly aerosol dataset.
WIF_DATA_RELEASE = "v1.0.1"

ASSET_URL_BASE_ENV = "WOOF_TABLE_ASSET_URL_BASE"

def asset_url_base() -> str:
    override = os.environ.get(ASSET_URL_BASE_ENV)
    return override.rstrip("/") if override else RELEASE_ASSET_BASE_URL


def wif_asset_url_base() -> str:
    """The fixed source archive or an explicitly selected raw-file mirror.

    This dataset is pinned independently of bridge binaries. Source installs
    and new releases must not need a published bridge release to acquire it.
    """

    override = os.environ.get(ASSET_URL_BASE_ENV)
    if override and override.strip():
        return override.strip().rstrip("/")

    override = os.environ.get("WOOF_BRIDGE_ASSET_URL_BASE")
    if override and override.strip():
        return override.strip().rstrip("/")
    from woof.ingest.wif_dataset import WIF_DATASET_DOWNLOAD_BASE

    return WIF_DATASET_DOWNLOAD_BASE


def wif_asset_source() -> tuple[str, str | None]:
    from woof.ingest.wif_dataset import WIF_DATASET_FILE

    mirrored = any(os.environ.get(key, "").strip()
                   for key in (ASSET_URL_BASE_ENV, "WOOF_BRIDGE_ASSET_URL_BASE"))
    return (wif_asset_url_base() + "/" + WIF_DATASET_FILE
            + ("" if mirrored else ".bz2"), None if mirrored else "bz2")


class TableAssetError(RuntimeError):
    """A refusal: wrong bytes, unfetchable asset, or unusable source."""


class MissingTableAssets(FileNotFoundError):
    """One-sentence refusal: a selected scheme's tables are not staged.

    A ``FileNotFoundError`` subclass on purpose.  Every stage that can
    meet this already turns ``OSError`` into a sentence rather than a
    traceback, so the sentence below reaches the reader through the
    handlers that exist instead of needing a new one at each site.
    """


def classify_assets(
        root: Path,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> tuple[list[TableAsset], list[str], list[TableAsset]]:
    """(valid, invalid-detail, absent) partition of ``assets`` at ``root``.

    ``invalid`` entries are human-readable refusals for files that exist
    with the wrong size or SHA-256; ``absent`` entries simply do not
    exist yet.
    """

    valid: list[TableAsset] = []
    invalid: list[str] = []
    absent: list[TableAsset] = []
    for asset in assets:
        path = root / asset.filename
        if not path.is_file():
            absent.append(asset)
            continue
        try:
            validate_table_assets(root, (asset,))
        except ValueError as error:
            invalid.append(str(error))
            continue
        valid.append(asset)
    return valid, invalid, absent


def missing_externalized_assets(
        root: Path,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> list[TableAsset]:
    """Externalized assets absent from ``root`` (fetchable gaps)."""

    _valid, _invalid, absent = classify_assets(root, assets)
    return [asset for asset in absent
            if asset.filename in EXTERNALIZED_TABLE_FILENAMES]


def unstaged_table_assets(
        root: Path | None = None,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> list[TableAsset]:
    """Assets that are simply not there in ``root``.

    Presence, and nothing else.  Size and SHA-256 stay where they
    belong -- re-validated at load, on every launch, by
    ``validate_table_assets``, which can name the expected and actual
    bytes.  Absence is the question a preflight owns, and answering it
    must not cost 362 MiB of hashing on every run; a file that is
    present but wrong is drift, a different sentence, and this function
    deliberately does not compete with it.
    """

    if root is None:
        from woof.physics_compat import thompson_table_root

        root = Path(thompson_table_root())
    root = Path(root)
    gaps: list[TableAsset] = []
    for asset in assets:
        try:
            present = (root / asset.filename).is_file()
        except OSError:
            present = False
        if not present:
            gaps.append(asset)
    return gaps


def _first_use_staged_assets() -> frozenset[TableAsset]:
    """Every pinned asset :func:`stage_classic_tables` puts in the root.

    The classic mp=8 set plus the mp=28 activation table
    (``CCN_ACTIVATE.BIN``), which the staging leg copies into the same
    root because the aerosol contract shares it.
    """

    from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

    return frozenset((*CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS))


def _thompson_scheme_label(assets: tuple[TableAsset, ...]) -> str:
    """``mp_physics=28 (aerosol-aware Thompson)`` or ``=8 (Thompson)``.

    Derived from the asset set the caller binds: only mp=28 binds the
    aerosol activation table.  The refusal used to hard-code ``8``, so an
    mp=28 run was told it was mp=8.
    """

    from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

    if any(asset in AEROSOL_TABLE_ASSETS for asset in assets):
        return "mp_physics=28 (aerosol-aware Thompson)"
    return "mp_physics=8 (Thompson)"


def auto_stage_applies(
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS) -> bool:
    """Will an incomplete root be staged on first use rather than refused?

    True when no operator named a root (``WOOF_THOMPSON_TABLE_ROOT``
    unset) and every asset in ``assets`` is one ``woof fetch-tables``
    stages, pins included: the classic mp=8 set, or the classic set plus
    the mp=28 activation table.  It used to demand exactly the classic
    set, so an mp=28 run on a fresh install (classic + CCN_ACTIVATE.BIN)
    skipped first-use staging and was refused at prepare, after go had
    already downloaded the forcing.  A named root is the operator's
    mirror and is never filled behind their back (the rule
    :func:`staging_is_self_chosen` states), and a caller binding its own
    table set (different pins) must not be handed this one.
    """

    from woof.physics_compat import THOMPSON_TABLE_ROOT_ENV

    return (not os.environ.get(THOMPSON_TABLE_ROOT_ENV)
            and bool(assets)
            and set(assets) <= _first_use_staged_assets())


def _stage_on_first_use(gaps: list[TableAsset],
                        scheme: str = "mp_physics=8 (Thompson)") -> str | None:
    """Run the ``woof fetch-tables`` classic leg; None, or why it failed.

    The defect this closes (2.9.2): a clean ``pip install recast-woof`` could
    not run ``mp_physics=8``.  The wheel cannot carry freezeH2O.dat and
    qr_acr_qg_V4.dat (PyPI's 100 MiB per-file cap; the recast-woof-data wheel
    is already 98.6 MB without them), so every fresh install, image
    build and new box refused its first Thompson forecast until someone
    ran ``woof fetch-tables`` by hand (the image build and the 2x MPAS
    box both hit it on 2026-10-16).  The fork Thompson set has been
    acquired on first use since 2.8.7; the classic set now is too,
    through the identical staging leg (same pinned sources, same SHA-256
    verification before the atomic install, same lock), into
    ``~/.woof/tables/thompson`` so it is paid for once per machine.
    Offline machines still stage with ``woof fetch-tables --from DIR``.
    The leg's progress goes to stderr so a door's stdout stays its own.
    """

    import argparse
    import contextlib

    mib = sum(asset.bytes for asset in gaps) / (1024 * 1024)
    print(f"woof: {scheme} first use on this machine: "
          f"staging {', '.join(asset.filename for asset in gaps)} "
          f"({mib:.0f} MiB, SHA-256 verified against the packaged pins) "
          f"into the user table root, once -- offline machines stage them "
          f"with `woof fetch-tables --from DIR`", file=sys.stderr,
          flush=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code = stage_classic_tables(
                argparse.Namespace(root=None, from_dir=None))
    except (TableAssetError, GeogFetchError, OSError, ValueError) as error:
        return str(error)
    return None if code == 0 else f"the fetch-tables leg exited {code}"


def require_thompson_tables(
        root: Path | None = None,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
        *,
        stage: bool = True,
) -> Path:
    """The resolved mp8 table root, staged on first use, or a refusal.

    THE preflight for ``mp_physics=8``.  It exists because the same
    condition used to be discovered by ``validate_table_assets`` at the
    top of the GPU run -- after the fetch, after preprocessing, minutes
    of paid work in -- and reached the reader as a ``FileNotFoundError``
    traceback naming a path inside site-packages.  Called from the
    stages that select physics, before anything expensive starts.

    When the root is resolved here (``root=None``) and
    :func:`auto_stage_applies`, an absent set is staged now by
    :func:`_stage_on_first_use` and the root re-resolved, so a fresh
    install runs Thompson with no manual step.  ``stage=False`` only
    inspects: when that same first-use route applies, its advisory
    describes the next run's automatic download.  A named root, a
    caller's own table set, or a staging failure instead names the
    manual command needed before preparing the forecast.
    """

    resolved_here = root is None
    scheme = _thompson_scheme_label(tuple(assets))
    if root is None:
        from woof.physics_compat import thompson_table_root

        root = Path(thompson_table_root())
    root = Path(root)
    # ``assets`` is a parameter because the pins are not universal: a
    # caller that binds its own table authority (and the harnesses that
    # stand in for one) must be checked against the set IT will load,
    # not against a second opinion this module happens to hold.
    gaps = unstaged_table_assets(root, assets)
    if not gaps:
        return root
    failure = None
    automatic_staging = resolved_here and auto_stage_applies(assets)
    if stage and automatic_staging:
        failure = _stage_on_first_use(gaps, scheme)
        if failure is None:
            from woof.physics_compat import thompson_table_root

            root = Path(thompson_table_root())
            gaps = unstaged_table_assets(root, assets)
            if not gaps:
                return root
    mib = sum(asset.bytes for asset in gaps) / (1024 * 1024)
    missing = (f"{scheme} needs "
               f"{', '.join(asset.filename for asset in gaps)}, which "
               f"{'are' if len(gaps) > 1 else 'is'} not staged at {root}")
    if not stage and automatic_staging:
        tables = ("the two tables" if len(gaps) == 2 else
                  "the missing table" if len(gaps) == 1 else
                  f"the {len(gaps)} missing tables")
        raise MissingTableAssets(
            f"{missing} -- the first `woof go` run downloads and checks "
            f"{tables} (about {mib:.0f} MiB) from the pinned release assets; "
            f"`woof fetch-tables` stages them ahead of time or offline "
            f"with `--from DIR`")
    # One sentence, so the failure's own text is folded onto one line.
    automatic = (" (automatic staging failed: "
                 + " ".join(failure.split()).replace(". ", "; ") + ")"
                 if failure else "")
    raise MissingTableAssets(
        f"{missing}{automatic} -- run "
        f"`woof fetch-tables` ({mib:.0f} MiB, SHA-256 verified against "
        f"the packaged pins; `--from DIR` offline) before preparing this "
        f"forecast")


def staging_is_self_chosen(root: Path) -> bool:
    """True when ``root`` is the user-level location THIS command picked.

    The distinction matters for one behaviour only: a self-chosen root
    is completed from the package (below), because nothing else will
    complete it and an incomplete root resolves to nothing.  A root an
    operator NAMED through ``WOOF_THOMPSON_TABLE_ROOT`` is theirs --
    this command fetches the externalized assets into it and reports
    anything else missing as a broken install, exactly as it always
    has, rather than quietly filling their mirror from the package.
    """

    from woof.physics_compat import (THOMPSON_TABLE_ROOT_ENV,
                                      user_thompson_table_root)

    if os.environ.get(THOMPSON_TABLE_ROOT_ENV):
        return False
    try:
        return Path(root) == user_thompson_table_root()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return False


def staging_root(root: Path | None = None) -> Path:
    """WHERE ``fetch-tables`` writes: the root the next run will read.

    Deliberately not "wherever ``thompson_table_root()`` points today".
    On a fresh wheel that is site-packages, which the next upgrade
    empties; the durable location is the user-level directory beside
    ``~/.woof/bridges``.  An install whose *packaged* root is already
    complete keeps it -- a clone, or a wheel staged before this change
    -- because that is the root :func:`thompson_table_root` resolves for
    it, and re-staging 362 MiB to change nothing is not a service.

    ``WOOF_THOMPSON_TABLE_ROOT`` still wins outright: an operator who
    named a root is telling this command where the bytes go.
    """

    if root is not None:
        return Path(root)
    from woof.physics_compat import (
        THOMPSON_TABLE_ROOT_ENV,
        packaged_thompson_table_root,
        user_thompson_table_root,
    )

    override = os.environ.get(THOMPSON_TABLE_ROOT_ENV)
    if override:
        return Path(override)
    packaged = packaged_thompson_table_root()
    if not unstaged_table_assets(packaged):
        return packaged
    try:
        return user_thompson_table_root()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return packaged


def _staging_path(root: Path, asset: TableAsset) -> Path:
    """A staging name only this process, and only this call, can own.

    The staging file used to be a fixed ``.<filename>.fetch-partial``
    shared by every writer into the table root.  Verification and the
    ``os.replace`` that installs the verified bytes are separate steps,
    so a second writer could rewrite that fixed file in between and get
    its bytes installed under the first writer's successful pin check --
    an install that never matched the pin it was blessed by.  A unique
    name means the bytes hashed are the bytes moved.

    The name was the pid and ``time.time_ns()``.  Windows' clock advances
    in coarse ticks, so writers in one process within one tick were given
    one name (50 names asked in a row were 1 name on the windows-2025 box),
    and two threads staging the same table wrote the same file.  A random
    UUID per call is unique without a clock or a lock.
    """

    return root / (f".{asset.filename}.fetch-partial-"
                   f"{os.getpid()}-{uuid.uuid4().hex}")


def _cleanup_interrupted(root: Path, asset: TableAsset) -> None:
    """Remove this asset's unpublished fragments only after their writers died.

    A terminated CLI cannot run its finally block. Its Python staging name
    and Rust range-transfer fragment would otherwise accumulate on every
    restart. Call under the root lock; never touch a live writer's files.
    """
    import re
    import psutil

    pattern = re.compile(r"^\." + re.escape(asset.filename)
                         + r"\.fetch-partial-(\d+)-[0-9a-f]{32}"
                         + r"(?:\.encoded)?(?:\.part-(\d+)-\d+)?$")
    def alive(pid):
        if pid is None:
            return False
        try:
            process = psutil.Process(int(pid))
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True
    removed = 0
    for path in root.glob(f".{asset.filename}.fetch-partial-*"):
        match = pattern.fullmatch(path.name)
        if (match is not None and not any(alive(pid) for pid in match.groups())
                and not path.is_symlink() and path.is_file()):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # A read-only cache can still hold a verified final table.
                # An untrusted fragment never participates in its validation.
                continue
            removed += 1
    if removed:
        print(f"woof fetch-tables: {asset.filename}: removed {removed} "
              "unpublished fragments from an interrupted transfer", file=sys.stderr)


def _stage(temp: Path, final: Path, asset: TableAsset) -> None:
    """Verify ``temp`` against ``asset`` pins, then atomically install."""

    actual_bytes = temp.stat().st_size
    if actual_bytes != asset.bytes:
        temp.unlink(missing_ok=True)
        raise TableAssetError(
            f"{asset.filename}: staged {actual_bytes} bytes, expected "
            f"{asset.bytes}; refused and deleted")
    actual_sha256 = _table_hash(temp)
    if actual_sha256 != asset.sha256:
        temp.unlink(missing_ok=True)
        raise TableAssetError(
            f"{asset.filename}: staged SHA-256 {actual_sha256}, expected "
            f"{asset.sha256}; refused and deleted")
    os.replace(temp, final)


def fetch_asset_from_url(root: Path, asset: TableAsset, url: str, *, compression=None) -> Path:
    rows = table_source_rows(asset)
    if any(row["url"] == url for row in rows):
        return fetch_asset_from_sources(root, asset)
    return _fetch_asset_from_url(root, asset, url, compression=compression)


def _fetch_asset_from_url(root: Path, asset: TableAsset, url: str, *, compression=None,
                          offline_command="woof fetch-tables --from DIR") -> Path:
    """Ranged Rust fetch and pin before the final atomic install."""
    if compression not in (None, "bz2", "zstd"):
        raise TableAssetError(f"{asset.filename}: unsupported compression {compression!r}; the raw table pin cannot be checked")
    root.mkdir(parents=True, exist_ok=True)
    final = root / asset.filename
    with fetch_guard.hold("fetch-tables", root):
        _cleanup_interrupted(root, asset)
        if final.is_file():
            _verify_in_place(final, asset)
            return final
        temp = _staging_path(root, asset)
        encoded = temp.with_name(temp.name + ".encoded") if compression else temp
        attrs = {"bytes": 0, "datasets": [asset.filename]}
        counter = ByteCounter(f"woof fetch-tables: {asset.filename}", asset.bytes)
        received_before = 0
        def progress(received, total):
            nonlocal received_before
            if total:
                counter.total = total
            counter.advance(max(0, received - received_before))
            received_before = received
        with span("prep.tables.fetch", attrs=attrs):
            try:
                if compression is None:
                    record = _rw_asset("asset", url=url, output=temp,
                                       bytes=asset.bytes, sha256=asset.sha256,
                                       on_progress=progress)
                    progress(record["bytes"], record["bytes"])
                    attrs["bytes"] = record["bytes"]
                    os.replace(temp, final)
                else:
                    record = _rw_asset("encoded-asset", url=url, output=encoded,
                                       on_progress=progress)
                    progress(record["bytes"], record["bytes"])
                    attrs["bytes"] = record["bytes"]
                    # rw_fetch decodes in process: the host bzip2 and pzstd
                    # this used to spawn are not on a stock Windows PATH,
                    # so every compressed table refused there.
                    _rw_asset("decode", input=encoded, output=temp, codec=compression)
                    _stage(temp, final, asset)
            except (GeogFetchError, OSError) as error:
                raise TableAssetError(
                    f"{asset.filename}: could not fetch {url}: {error}; "
                    f"offline, stage the pinned files with `{offline_command}`") from error
            finally:
                counter.close()
                temp.unlink(missing_ok=True)
                encoded.unlink(missing_ok=True)
        _cleanup_interrupted(root, asset)
    return final


def fetch_asset_from_dir(root: Path, asset: TableAsset,
                         source_dir: Path, *, replace_damaged: bool = False) -> Path:
    """Copy a pinned asset, optionally repairing owned damaged staging."""

    source = source_dir / asset.filename
    if not source.is_file():
        raise TableAssetError(
            f"{asset.filename}: not found in --from directory {source_dir}")
    final = root / asset.filename
    with fetch_guard.hold("fetch-tables", root):
        _cleanup_interrupted(root, asset)
        if final.is_file():
            try:
                _verify_in_place(final, asset)
            except TableAssetError:
                if not replace_damaged:
                    raise
            else:
                return final
        temp = _staging_path(root, asset)
        try:
            shutil.copyfile(source, temp)
        except OSError as error:
            temp.unlink(missing_ok=True)
            raise TableAssetError(
                f"{asset.filename}: copy from {source} failed: {error}")
        _stage(temp, final, asset)
    return final


def _packaged_copy_is_valid(packaged: Path, asset: TableAsset) -> bool:
    try:
        validate_table_assets(packaged, (asset,))
    except (OSError, ValueError):
        return False
    return True


def stage_classic_tables(args) -> int:
    """Stage the mandatory classic coefficient tables.  0, or 2 on refusal.

    The mandatory leg of ``fetch-tables``: every ``mp_physics=8`` and
    ``mp_physics=28`` run reads this set, and no flag makes it optional.
    Split out of :func:`fetch_tables_main` so the optional ``--wif`` leg
    cannot decide whether it runs.
    """

    from woof.physics_compat import packaged_thompson_table_root

    root = staging_root(getattr(args, "root", None))
    packaged = packaged_thompson_table_root()
    if not root.is_dir():
        if root == packaged:
            print(f"woof fetch-tables: table root {root} is not a "
                  "directory (broken install, or "
                  "WOOF_THOMPSON_TABLE_ROOT points nowhere)")
            return 2
        # The user-level staging root is ours to create, exactly as
        # `woof fetch-bridges` creates ~/.woof/bridges.
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"woof fetch-tables: cannot create table root {root}: "
                  f"{error}")
            return 2
        print(f"woof fetch-tables: staging into {root} (outside the "
              "install, so a wheel upgrade cannot delete it)")

    from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

    # Every table already in the root is pin-checked, the mp=28
    # activation table included: it used to be skipped whenever a file of
    # its name existed, so an empty CCN_ACTIVATE.BIN passed setup and
    # this command's "verified" summary and failed the first mp=28
    # forecast at load.  In the root this command owns, a damaged copy of
    # a table the package carries is replaced by the verified packaged
    # bytes; anywhere else it is refused, because an operator's mirror is
    # theirs to repair.
    self_chosen = root != packaged and staging_is_self_chosen(root)
    repaired: list[TableAsset] = []
    refusals: list[str] = []
    for asset in (*CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS):
        if not (root / asset.filename).is_file():
            continue
        try:
            validate_table_assets(root, (asset,))
            continue
        except (ValueError, OSError) as error:
            damage = str(error)
        if self_chosen and _packaged_copy_is_valid(packaged, asset):
            print(f"woof fetch-tables: {asset.filename} in {root} is "
                  f"damaged ({damage}); replacing it with the verified "
                  "copy from the package")
            try:
                # This branch already proved self-chosen user staging and a
                # valid packaged replacement. Keep the damaged file until a
                # pinned temp is ready for atomic replacement.
                fetch_asset_from_dir(root, asset, packaged, replace_damaged=True)
            except TableAssetError as error:
                refusals.append(str(error))
                continue
            repaired.append(asset)
            continue
        refusals.append(damage)
    for line in refusals:
        print(f"woof fetch-tables: REFUSED: {line} -- an existing file "
              "is never overwritten; delete it and re-run")
    if refusals:
        return 2

    valid, invalid, absent = classify_assets(root)

    # A staged root has to hold the WHOLE set: resolution asks whether a
    # root is complete, so downloading the two externalized assets into
    # ~/.woof and leaving the other two behind in site-packages would
    # produce two half-roots and no runnable one.  Assets the package
    # does carry are copied across under the same pin verification.
    copied: list[TableAsset] = []
    if root != packaged:
        available = {asset.filename for asset in classify_assets(packaged)[0]}
        for asset in list(absent):
            if asset.filename in EXTERNALIZED_TABLE_FILENAMES and not self_chosen:
                continue
            if asset.filename not in available:
                continue
            mib = asset.bytes / (1024 * 1024)
            print(f"woof fetch-tables: copying {asset.filename} "
                  f"({mib:.1f} MiB) from the package into {root}")
            try:
                fetch_asset_from_dir(root, asset, packaged)
            except TableAssetError as error:
                print(f"woof fetch-tables: REFUSED: {error}")
                return 2
            absent.remove(asset)
            copied.append(asset)
        # mp=28 reads CCN_ACTIVATE.BIN out of the SAME root as the
        # classic tables -- the aerosol contract shares the root on
        # purpose, so tnccn_act and the process tables cannot come from
        # two different WRF builds.  It is not a member of
        # CLASSIC_TABLE_ASSETS, so the loop above never carried it, and
        # a staged root without it is complete for mp=8 and fatal for
        # mp=28: on a wheel install `fetch-tables` reported the root
        # "complete and byte-valid" and the next mp=28 forecast died at
        # its first microphysics step with MissingAerosolTableAsset,
        # while the packaged copy sat in site-packages the whole time.
        # Copied under the identical pin verification; a copy already
        # present was pin-checked (and repaired) above.
        for asset in AEROSOL_TABLE_ASSETS:
            if (root / asset.filename).is_file():
                continue
            if not (packaged / asset.filename).is_file():
                continue
            print(f"woof fetch-tables: copying {asset.filename} "
                  f"({asset.bytes:,} B, the mp=28 activation table) from "
                  f"the package into {root}")
            try:
                fetch_asset_from_dir(root, asset, packaged)
            except TableAssetError as error:
                print(f"woof fetch-tables: REFUSED: {error}")
                return 2
            copied.append(asset)

    declared = {asset.filename: table_source_rows(asset) for asset in absent}
    packaged_gaps = [asset for asset in absent
                     if asset.filename not in EXTERNALIZED_TABLE_FILENAMES
                     and not declared[asset.filename]]
    for asset in packaged_gaps:
        print(f"woof fetch-tables: {asset.filename} is missing but ships "
              "inside the package itself; this install is incomplete -- "
              "reinstall woof rather than fetching")
    if packaged_gaps:
        return 2

    fetchable = [asset for asset in absent
                 if asset.filename in EXTERNALIZED_TABLE_FILENAMES
                 or declared[asset.filename]]
    if not fetchable and not copied:
        if repaired:
            validate_table_assets(root)
            print(f"woof fetch-tables: table root {root} repaired "
                  f"({', '.join(a.filename for a in repaired)}) and "
                  "byte-valid")
            return 0
        print(f"woof fetch-tables: all {len(valid)} table assets at "
              f"{root} verified (exact size + SHA-256); nothing to fetch")
        return 0

    source_dir = getattr(args, "from_dir", None)
    base = asset_url_base()
    for asset in fetchable:
        mib = asset.bytes / (1024 * 1024)
        try:
            if source_dir is not None:
                print(f"woof fetch-tables: staging {asset.filename} "
                      f"({mib:.1f} MiB) from {source_dir}")
                final = fetch_asset_from_dir(root, asset, Path(source_dir))
            else:
                url = f"{base}/{asset.filename}"
                print(f"woof fetch-tables: downloading {asset.filename} "
                      f"({mib:.1f} MiB) from {url}")
                final = fetch_asset_from_sources(root, asset)
        except TableAssetError as error:
            print(f"woof fetch-tables: REFUSED: {error}")
            return 2
        print(f"woof fetch-tables: verified and installed {final} "
              f"({asset.bytes:,} B, SHA-256 {asset.sha256[:12]}...)")

    # Close the loop with the exact validation every mp8 run performs.
    validate_table_assets(root)
    print(f"woof fetch-tables: table root {root} complete and byte-valid")
    return 0


def fetch_tables_main(args) -> int:
    """Run the mandatory table leg and the optional ``--wif`` leg.

    The two legs are INDEPENDENT.  ``--wif`` names an extra optional
    dataset; it does not change whether the coefficient tables every run
    needs get staged.  This used to run the WIF leg first and return on
    its exit code, so `woof fetch-tables --wif` against an unreachable
    dataset refused and left the mandatory tables unstaged -- the user
    asked for both and got neither, with nothing in the output saying the
    classic leg had never been attempted.

    The exit code is the worst of the legs that ran, and the closing
    summary names each leg as staged or refused, so "did my tables land"
    is answered by reading the last line rather than by inferring it from
    a single number.
    """

    schemes = getattr(args, "schemes", None)
    if getattr(args, "replace_damaged", False) and (not schemes or not getattr(args, "from_dir", None)):
        print("woof fetch-tables: --replace-damaged requires --schemes and --from DIR "
              "so the replacement source is explicit", file=sys.stderr)
        return 2
    if schemes:
        if any(getattr(args, key, False) for key in
               ("wif", "wif_only", "ocean", "ocean_only", "thompson_fork", "thompson_fork_only")):
            print("woof fetch-tables: --schemes selects its own table set; "
                  "run additional dataset legs in a separate fetch-tables command", file=sys.stderr)
            return 2
        from woof.tables import stage_schemes
        try:
            report = stage_schemes(schemes, source_dir=getattr(args, "from_dir", None),
                                   replace_damaged=getattr(args, "replace_damaged", False))
        except (TableAssetError, OSError, ValueError) as error:
            print(f"woof fetch-tables: REFUSED: {error}", file=sys.stderr)
            return 2
        print(f"woof fetch-tables: {len(report.present)} required files verified (size + SHA-256)")
        return 0
    wif_requested = bool(getattr(args, "wif", False))
    wif_only = bool(getattr(args, "wif_only", False))
    fork_requested = bool(getattr(args, "thompson_fork", False))
    fork_only = bool(getattr(args, "thompson_fork_only", False))
    fork_code: int | None = None
    if fork_requested or fork_only:
        from woof.thompson_fork_assets import stage_thompson_fork_tables
        fork_code = stage_thompson_fork_tables(
            getattr(args, "from_dir", None), getattr(args, "thompson_fork_root", None))

    ocean_requested = bool(getattr(args, "ocean", False))
    ocean_only = bool(getattr(args, "ocean_only", False))
    ocean_code: int | None = None
    if ocean_requested or ocean_only:
        ocean_code = stage_ocean_climatology(getattr(args, "from_dir", None),
                                             getattr(args, "ocean_root", None))

    wif_code: int | None = None
    if wif_requested:
        wif_code = stage_wif_dataset(getattr(args, "from_dir", None),
                                     getattr(args, "wif_root", None))

    # --wif-only is the one flag that legitimately skips the mandatory
    # leg, because it is the operator saying so in as many words.
    classic_code: int | None = None
    if not wif_only and not fork_only and not ocean_only:
        classic_code = stage_classic_tables(args)

    legs = []
    if classic_code is not None:
        legs.append(("classic coefficient tables", classic_code))
    if wif_code is not None:
        legs.append(("--wif aerosol climatology dataset", wif_code))
    if fork_code is not None:
        legs.append(("--thompson-fork coefficient tables", fork_code))
    if ocean_code is not None:
        legs.append(("--ocean WOA18 mixed-layer-depth climatology", ocean_code))
    if len(legs) > 1 or (wif_requested and not wif_only):
        for label, code in legs:
            state = "staged" if code == 0 else "REFUSED"
            print(f"woof fetch-tables: {label}: {state}")
    return max((code for _, code in legs), default=0)


def stage_ocean_climatology(source_dir=None, root=None) -> int:
    """Stage all 12 NOAA WOA18 monthly mixed-layer-depth files.

    The mixed-layer ocean is on by default for forecasts of 48 h and
    more, and its initial mixed-layer depth is this climatology.  Each
    month is SHA-256 verified against
    :data:`woof.ocean.mixed_layer.climatology.MONTH_PINS` before the
    atomic install, into the very cache the forecast reads
    (``WOOF_OCEAN_CACHE``, else ``<case data root>/ocean``), so an image
    or AMI baked with this command never touches NCEI at forecast start.
    Named breakage: NCEI THREDDS answered 503 for over an hour on
    2026-10-22 and every run with water in its domain died at its first
    step, because nothing could stage the file ahead of time.
    """

    from woof.ocean.mixed_layer import climatology

    target = climatology.cache_root(root)
    print(f"woof fetch-tables --ocean: staging 12 WOA18 monthly MLD files "
          f"into {target}" + (f" from {source_dir}" if source_dir else ""))
    staged, failed = climatology.stage_all(cache=target, source_dir=source_dir)
    if failed:
        print(f"woof fetch-tables --ocean: {len(staged)} of 12 months staged; "
              f"months {', '.join(f'{m:02d}' for m in failed)} REFUSED -- re-run "
              "when the source answers, or stage them with --from DIR")
        return 2
    print(f"woof fetch-tables --ocean: all 12 months staged and verified at {target}")
    return 0


def stage_wif_dataset(source_dir=None, root=None) -> int:
    """Stage the mp=28 WIF climatology dataset under the table contract.

    The bytes are verified against
    :data:`woof.ingest.wif_dataset.WIF_DATASET_ASSET` -- exact size then
    SHA-256 -- BEFORE the atomic move, an existing file with the wrong
    bytes is refused rather than overwritten, and a present, valid file
    costs one hash and exits 0.  That is ``_stage``'s contract unchanged;
    only the root and the pin differ, because the reason this file is
    externalized is the same reason ``freezeH2O.dat`` is.
    """

    from woof.ingest.wif_dataset import (
        WIF_DATASET_ASSET, WIF_DATASET_FILE, resolve_wif_data_root)

    wif_root = resolve_wif_data_root(root)
    try:
        wif_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"woof fetch-tables --wif: cannot create WIF data root "
              f"{wif_root}: {error}")
        return 2
    final = wif_root / WIF_DATASET_FILE
    mib = WIF_DATASET_ASSET.bytes / (1024 * 1024)
    if final.is_file():
        try:
            _verify_in_place(final, WIF_DATASET_ASSET)
        except TableAssetError as error:
            print(f"woof fetch-tables --wif: REFUSED: {error} -- an "
                  "existing file is never overwritten; delete it and re-run")
            return 2
        print(f"woof fetch-tables --wif: {final} already staged and "
              f"byte-valid ({WIF_DATASET_ASSET.bytes:,} B, SHA-256 "
              f"{WIF_DATASET_ASSET.sha256[:12]}...); nothing to fetch")
        return 0
    try:
        if source_dir is not None:
            print(f"woof fetch-tables --wif: staging {WIF_DATASET_FILE} "
                  f"({mib:.1f} MiB) from {source_dir}")
            fetch_asset_from_dir(wif_root, WIF_DATASET_ASSET,
                                 Path(source_dir))
        else:
            url, compression = wif_asset_source()
            print(f"woof fetch-tables --wif: downloading "
                  f"{WIF_DATASET_FILE} ({mib:.1f} MiB) from {url}")
            fetch_asset_from_sources(wif_root, WIF_DATASET_ASSET)
    except TableAssetError as error:
        print(f"woof fetch-tables --wif: REFUSED: {error}")
        return 2
    print(f"woof fetch-tables --wif: verified and installed {final} "
          f"({WIF_DATASET_ASSET.bytes:,} B, SHA-256 "
          f"{WIF_DATASET_ASSET.sha256[:12]}...)")
    return 0


def _table_hash(path: Path) -> str:
    try:
        return _rw_asset("hash", input=path)["sha256"]
    except GeogFetchError as error:
        raise TableAssetError(f"{path}: {error}") from error


def _verify_in_place(path: Path, asset: TableAsset) -> None:
    """Size then SHA-256 of a file already in its final location."""

    actual_bytes = path.stat().st_size
    if actual_bytes != asset.bytes:
        raise TableAssetError(
            f"{path}: {actual_bytes} bytes, expected {asset.bytes}")
    digest = _table_hash(path)
    if digest != asset.sha256:
        raise TableAssetError(
            f"{path}: SHA-256 {digest}, expected {asset.sha256}; other lookup coefficients would change the selected physics")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch-tables",
        help="download the externalized physics table assets (currently "
             "the 243 MiB Thompson freezeH2O.dat) into ~/.woof/tables/"
             "thompson -- outside the install, so a wheel upgrade cannot "
             "delete them -- SHA-256-verified against the packaged pins "
             "before install; idempotent when everything is already "
             "present")
    parser.add_argument("--schemes", nargs="+", metavar="FAMILY",
                        help="stage only named physics table families, e.g. p3 noah tempo rrtmgp; see woof.tables")
    parser.add_argument("--replace-damaged", action="store_true",
                        help="with --schemes and --from, replace damaged files only after source pin verification")
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage from a local directory instead of downloading "
             "(offline installs); verification is identical")
    parser.add_argument(
        "--wif", action="store_true",
        help="also stage QNWFA_QNIFA_SIGMA_MONTHLY.dat (215 MiB), the "
             "global monthly aerosol climatology the mp_physics=28 WIF "
             "ingest reads (aer_init_opt=1 with wif_input_opt=1), into "
             "~/.woof/wif under the same SHA-256 contract.  Opt-in: it "
             "is an input dataset, not a coefficient table. Forecast "
             "fetches acquire it automatically when selected physics needs it")
    parser.add_argument(
        "--wif-only", dest="wif_only", action="store_true",
        help="with --wif, stage only that dataset and leave the "
             "coefficient tables alone")
    parser.add_argument(
        "--wif-root", dest="wif_root", metavar="DIR", default=None,
        help="stage the WIF dataset into DIR instead of ~/.woof/wif "
             "(same meaning as WOOF_WIF_DATA_ROOT)")
    parser.add_argument(
        "--thompson-fork", action="store_true",
        help="also stage the pinned WRF 3.9 fork Thompson coefficient set "
             "(thompson_version = \"wrf_39_noaa\", the HRRR configurations' "
             "tables; 330 MiB) into ~/.woof/tables/thompson-wrf39-noaa: from "
             "--from DIR, packaged fork data, an explicitly selected mirror, or "
             "the release assets, SHA-256-verified against the packaged pins "
             "before install. Runs that select the fork acquire it automatically")
    parser.add_argument(
        "--thompson-fork-only", action="store_true",
        help="stage only the fork coefficient set and leave classic tables alone")
    parser.add_argument(
        "--thompson-fork-root", metavar="DIR", default=None,
        help="stage fork tables into DIR instead of the selected fork cache")
    parser.add_argument(
        "--ocean", action="store_true",
        help="also stage the 12 NOAA WOA18 monthly mixed-layer-depth files "
             "(about 85 MB) the default mixed-layer ocean initializes from, "
             "SHA-256 verified, into the cache forecasts read "
             "(WOOF_OCEAN_CACHE, else <case data root>/ocean), so no "
             "forecast fetches them from NCEI at start")
    parser.add_argument(
        "--ocean-only", action="store_true",
        help="stage only the ocean climatology and leave the coefficient "
             "tables alone")
    parser.add_argument(
        "--ocean-root", metavar="DIR", default=None,
        help="stage the ocean climatology into DIR (same meaning as "
             "WOOF_OCEAN_CACHE)")
    parser.set_defaults(func=fetch_tables_main)
    return parser


__all__ = [
    "ASSET_URL_BASE_ENV",
    "EXTERNALIZED_TABLE_FILENAMES",
    "MissingTableAssets",
    "RELEASE_ASSET_BASE_URL",
    "TableAssetError",
    "asset_url_base",
    "auto_stage_applies",
    "classify_assets",
    "fetch_asset_from_dir",
    "fetch_asset_from_url",
    "fetch_tables_main",
    "missing_externalized_assets",
    "register_cli",
    "require_thompson_tables",
    "staging_is_self_chosen",
    "stage_classic_tables",
    "stage_ocean_climatology",
    "stage_wif_dataset",
    "staging_root",
    "unstaged_table_assets",
    "wif_asset_url_base",
    "wif_asset_source",
]

def generated_table_root(name: str) -> Path:
    """Persistent cache for WOOF-built, manifest-pinned coefficient assets."""
    import re
    if re.fullmatch(r"[a-z][a-z0-9-]*", name) is None:
        raise ValueError("generated table family must be a plain identifier")
    base=os.environ.get("WOOF_GENERATED_TABLE_ROOT")
    return (Path(base) if base else Path.home()/".woof"/"tables")/name
