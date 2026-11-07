"""``woof fetch-tables``: staging, verification, and refusal contract.

The externalized Thompson asset (freezeH2O.dat) ships as a release
asset, not in the wheel or repository, so the fetch path IS the install
path for it: every byte must be verified against the thompson_contract
pins before an atomic install, mismatches must be refused and deleted,
and an existing wrong file must never be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pytest

from woof import table_assets
from woof.core.thompson_contract import (
    CLASSIC_TABLE_ASSETS,
    TableAsset,
    validate_table_assets,
)
from woof.physics_compat import packaged_thompson_table_root

REPO_ROOT = Path(__file__).resolve().parents[1]


def _asset_for(payload: bytes, name: str = "synthetic.dat") -> TableAsset:
    return TableAsset(name, len(payload),
                      hashlib.sha256(payload).hexdigest())


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**{"from_dir": None, **kwargs})


# ---------------------------------------------------------------------------
# Unit: staging semantics with synthetic assets
# ---------------------------------------------------------------------------

def test_fetch_from_dir_verifies_then_installs_atomically(tmp_path):
    payload = b"\x00\x01" * 4096
    asset = _asset_for(payload)
    source = tmp_path / "src"
    root = tmp_path / "root"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(payload)

    final = table_assets.fetch_asset_from_dir(root, asset, source)
    assert final == root / asset.filename
    assert final.read_bytes() == payload
    # no partial droppings
    assert list(root.glob(".*fetch-partial")) == []


def test_fetch_from_dir_refuses_and_deletes_wrong_bytes(tmp_path):
    payload = b"\x00\x01" * 4096
    asset = _asset_for(payload)
    source = tmp_path / "src"
    root = tmp_path / "root"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(b"corrupted" + payload)

    with pytest.raises(table_assets.TableAssetError, match="refused"):
        table_assets.fetch_asset_from_dir(root, asset, source)
    assert not (root / asset.filename).exists()
    assert list(root.glob(".*fetch-partial")) == []


def test_damaged_repair_preserves_original_until_replacement_is_verified(tmp_path):
    payload = b"verified replacement"
    asset = _asset_for(payload)
    source, root = tmp_path / "source", tmp_path / "root"
    source.mkdir(), root.mkdir()
    final = root / asset.filename
    final.write_bytes(b"damaged")
    (source / asset.filename).write_bytes(b"wrong replacement")

    with pytest.raises(table_assets.TableAssetError, match="refused"):
        table_assets.fetch_asset_from_dir(root, asset, source, replace_damaged=True)
    assert final.read_bytes() == b"damaged"
    assert not list(root.glob(".*fetch-partial-*"))

    (source / asset.filename).write_bytes(payload)
    with pytest.raises(table_assets.TableAssetError):
        table_assets.fetch_asset_from_dir(root, asset, source)
    assert final.read_bytes() == b"damaged"
    assert table_assets.fetch_asset_from_dir(root, asset, source, replace_damaged=True) == final
    assert final.read_bytes() == payload
    assert not list(root.glob(".*fetch-partial-*"))


def test_writers_in_one_process_and_one_clock_tick_never_share_a_staging_name(tmp_path):
    """Threads that ask for a staging name together each get their own.

    The name was the pid and ``time.time_ns()``; Windows' clock advances in
    coarse ticks, so 50 names asked in a row were one name on the
    windows-2025 box, and two threads staging one table would have written
    the same file -- the race a unique staging name exists to close.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    asset = _asset_for(b"x")
    root = tmp_path / "root"
    go = threading.Barrier(8)

    def names(_):
        go.wait(timeout=10)
        return [table_assets._staging_path(root, asset).name for _ in range(50)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        every = [name for batch in pool.map(names, range(8)) for name in batch]
    assert len(set(every)) == len(every) == 400
    prefix = f".{asset.filename}.fetch-partial-{os.getpid()}-"
    assert all(name.startswith(prefix) for name in every)


@pytest.mark.parametrize("copy", [b"bad", b"mismatch"])
def test_packaged_repair_preserves_the_damaged_target_if_source_pin_fails(
        tmp_path, copy):
    asset = _asset_for(b"verified")
    source = tmp_path / "package"
    root = tmp_path / "staged"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(copy)
    damaged = root / asset.filename
    damaged.write_bytes(b"")

    with pytest.raises(table_assets.TableAssetError, match="refused"):
        table_assets.fetch_asset_from_dir(root, asset, source, replace_damaged=True)
    assert damaged.read_bytes() == b""
    assert list(root.iterdir()) == [damaged]


def test_fetch_from_url_verifies_sha256_not_just_size(tmp_path):
    payload = os.urandom(8192)
    asset = _asset_for(payload)
    served = tmp_path / "served"
    root = tmp_path / "root"
    served.mkdir(), root.mkdir()

    # same size, different bytes -> refusal
    evil = bytearray(payload)
    evil[0] ^= 0xFF
    (served / asset.filename).write_bytes(bytes(evil))
    url = (served / asset.filename).as_uri()
    with pytest.raises(table_assets.TableAssetError, match="SHA-256"):
        table_assets.fetch_asset_from_url(root, asset, url)
    assert not (root / asset.filename).exists()

    # correct bytes -> installed
    (served / asset.filename).write_bytes(payload)
    final = table_assets.fetch_asset_from_url(root, asset, url)
    assert final.read_bytes() == payload


def test_classify_assets_partitions_valid_invalid_absent(tmp_path):
    good = b"good" * 1000
    asset_ok = _asset_for(good, "ok.dat")
    asset_bad = _asset_for(b"expected", "bad.dat")
    asset_gone = _asset_for(b"absent", "gone.dat")
    (tmp_path / "ok.dat").write_bytes(good)
    (tmp_path / "bad.dat").write_bytes(b"not-expected")

    valid, invalid, absent = table_assets.classify_assets(
        tmp_path, (asset_ok, asset_bad, asset_gone))
    assert [a.filename for a in valid] == ["ok.dat"]
    assert len(invalid) == 1 and "bad.dat" in invalid[0]
    assert [a.filename for a in absent] == ["gone.dat"]


# ---------------------------------------------------------------------------
# Integration: the real pins, the real packaged root
# ---------------------------------------------------------------------------

def _staged_root_without_externalized(tmp_path, monkeypatch) -> Path:
    """A table root holding the packaged (wheel-shipped) assets only.

    This is exactly what a wheel install looks like before
    ``woof fetch-tables`` runs.  Hardlinks keep the copies free.
    """

    packaged = packaged_thompson_table_root()
    if not packaged.is_dir():
        pytest.skip("packaged table root absent (wheel-only checkout)")
    root = tmp_path / "tables"
    root.mkdir()
    for asset in CLASSIC_TABLE_ASSETS:
        if asset.filename in table_assets.EXTERNALIZED_TABLE_FILENAMES:
            continue
        source = packaged / asset.filename
        if not source.is_file():
            pytest.skip(f"packaged asset absent: {asset.filename}")
        try:
            os.link(source, root / asset.filename)
        except OSError:
            import shutil
            shutil.copyfile(source, root / asset.filename)
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(root))
    return root


def test_doctor_names_the_fetch_remedy_for_the_externalized_gap(
        tmp_path, monkeypatch):
    from woof import doctor

    _staged_root_without_externalized(tmp_path, monkeypatch)
    check = doctor._thompson_tables_check()
    assert check.status == "missing"
    assert "freezeH2O.dat" in check.detail
    assert "externalized" in check.detail
    assert check.remedy is not None
    assert "woof fetch-tables" in check.remedy
    # the reinstall hint is the wrong remedy for a fetchable gap
    assert "reinstall" not in check.remedy


def test_fetch_tables_main_stages_offline_and_is_idempotent(
        tmp_path, monkeypatch, capsys):
    packaged = packaged_thompson_table_root()
    externalized = [packaged / name
                    for name in table_assets.EXTERNALIZED_TABLE_FILENAMES]
    if not all(path.is_file() for path in externalized):
        pytest.skip("externalized asset bytes not present in this checkout")

    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(packaged))) == 0
    validate_table_assets(root)  # the exact load-time validation
    out = capsys.readouterr().out
    assert "verified and installed" in out

    # second run: nothing to fetch, still 0, no network, no rewrite
    before = (root / "freezeH2O.dat").stat().st_mtime_ns
    assert table_assets.fetch_tables_main(_args()) == 0
    assert (root / "freezeH2O.dat").stat().st_mtime_ns == before
    assert "nothing to fetch" in capsys.readouterr().out


def test_fetch_tables_main_refuses_existing_wrong_bytes(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "freezeH2O.dat").write_bytes(b"tampered")
    assert table_assets.fetch_tables_main(_args()) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "never overwritten" in out
    # refusal must not delete or replace the operator's file
    assert (root / "freezeH2O.dat").read_bytes() == b"tampered"


def test_fetch_tables_main_keeps_refusal_for_a_gap_without_source_rows(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "thompson_aux_tables.dat").unlink()
    from woof import physics_compat
    empty_package = tmp_path / "empty-package"
    empty_package.mkdir()
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root", lambda: empty_package)
    monkeypatch.setattr(table_assets, "table_source_rows", lambda asset: ())
    assert table_assets.fetch_tables_main(_args()) == 2
    assert "reinstall" in capsys.readouterr().out


def test_fetch_tables_main_fills_a_declared_packaged_gap_in_an_explicit_root(
        tmp_path, monkeypatch):
    packaged = packaged_thompson_table_root()
    if not all((packaged / asset.filename).is_file() for asset in CLASSIC_TABLE_ASSETS):
        pytest.skip("complete pinned companion fixture is absent")
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "thompson_aux_tables.dat").unlink()
    assert table_assets.fetch_tables_main(_args(from_dir=str(packaged))) == 0
    validate_table_assets(root)


def test_cli_dispatches_fetch_tables_for_real(monkeypatch, capsys):
    """Through woof.cli.main, not just --help: the dispatch table must
    route fetch-tables to its handler (a --help-only probe once passed
    while the real dispatch raised AttributeError)."""

    from woof.cli import main

    packaged = packaged_thompson_table_root()
    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(packaged))
    assert main(["fetch-tables"]) == 0
    assert "nothing to fetch" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The staging location: outside the install, or a wheel upgrade deletes it
# ---------------------------------------------------------------------------

def _wheel_shaped_install(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """(packaged root missing the externalized pair, user-level root).

    What a fresh ``pip install recast-woof`` looks like: the two assets under
    PyPI's per-file cap are inside site-packages, the two over it are
    not anywhere yet.  ``~/.woof`` is redirected into ``tmp_path`` so
    the test never touches the real home directory.
    """

    real_packaged = packaged_thompson_table_root()
    if not real_packaged.is_dir():
        pytest.skip("packaged table root absent (wheel-only checkout)")
    packaged = tmp_path / "site-packages" / "woof" / "data" / "thompson"
    packaged = packaged / "tables"
    packaged.mkdir(parents=True)
    for asset in CLASSIC_TABLE_ASSETS:
        if asset.filename in table_assets.EXTERNALIZED_TABLE_FILENAMES:
            continue
        source = real_packaged / asset.filename
        if not source.is_file():
            pytest.skip(f"packaged asset absent: {asset.filename}")
        try:
            os.link(source, packaged / asset.filename)
        except OSError:
            import shutil
            shutil.copyfile(source, packaged / asset.filename)

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        "woof.physics_compat.packaged_thompson_table_root",
        lambda: packaged)
    return packaged, home / ".woof" / "tables" / "thompson"


def test_a_wheel_stages_outside_site_packages_and_resolves_there(
        tmp_path, monkeypatch, capsys):
    """`FileNotFoundError: missing Thompson table asset
    .../site-packages/woof/data/thompson/tables/qr_acr_qg_V4.dat`

    A wheel user paid for a 315 MiB `woof fetch-tables`, then a wheel
    upgrade deleted every byte of it, because staging went INSIDE
    site-packages.  Staging now lands beside ~/.woof/bridges, and the
    resolver reads it there.
    """

    from woof.physics_compat import thompson_table_root

    real_packaged = packaged_thompson_table_root()
    # Same guard as the offline staging test above, and for the same
    # reason: this stages `--from` the packaged root, and a published
    # clone deliberately does not carry the externalized assets --
    # `woof fetch-tables` downloads them.  Without this the release
    # snapshot fails its own suite for doing exactly what it should.
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)

    # Before staging: the resolver can only answer with the packaged
    # root, and it is short the two externalized assets.
    assert Path(thompson_table_root()) == packaged
    assert {a.filename for a in table_assets.unstaged_table_assets()} == set(
        table_assets.EXTERNALIZED_TABLE_FILENAMES)

    assert table_assets.staging_root() == user_root
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    printed = capsys.readouterr().out
    assert "so a wheel upgrade cannot delete it" in printed

    # The staged root is COMPLETE -- half of it in site-packages would
    # resolve to nothing -- and it is what a run now reads.
    validate_table_assets(user_root)
    assert Path(thompson_table_root()) == user_root
    assert table_assets.unstaged_table_assets() == []

    # The wheel upgrade that used to erase the work: site-packages is
    # emptied, and the staged set still answers.
    for path in packaged.iterdir():
        path.unlink()
    assert Path(thompson_table_root()) == user_root
    assert table_assets.unstaged_table_assets() == []


def test_a_complete_packaged_root_is_left_alone(tmp_path, monkeypatch,
                                                capsys):
    """Negative control: a clone, and a wheel staged before this change.

    Both have all four assets in the packaged root, and neither should
    be told to re-download 362 MiB into a new location.
    """

    from woof.physics_compat import thompson_table_root

    packaged = packaged_thompson_table_root()
    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert Path(thompson_table_root()) == packaged
    assert table_assets.staging_root() == packaged
    assert table_assets.fetch_tables_main(_args()) == 0
    assert "nothing to fetch" in capsys.readouterr().out
    assert not (home / ".woof").exists()


def test_a_named_root_is_never_filled_from_the_package(tmp_path,
                                                       monkeypatch):
    """WOOF_THOMPSON_TABLE_ROOT stays the operator's to populate.

    The completion-from-package step exists so a root THIS command
    chose is whole.  A mirror an operator named is not that, and
    quietly copying the package into it would make their explicit
    configuration mean something else.
    """

    root = tmp_path / "mirror"
    root.mkdir()
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(root))
    assert table_assets.staging_root() == root
    assert table_assets.staging_is_self_chosen(root) is False


# ---------------------------------------------------------------------------
# The refusal: one sentence at preflight, never a traceback mid-forecast
# ---------------------------------------------------------------------------

def test_a_missing_table_is_one_sentence_naming_the_file_and_the_command(
        tmp_path, monkeypatch):
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(root))
    with pytest.raises(table_assets.MissingTableAssets) as raised:
        table_assets.require_thompson_tables()
    message = str(raised.value)
    # One sentence: no line breaks, and no full stop that starts
    # another one.  (Filenames carry dots; sentences carry ". ".)
    assert "\n" not in message
    assert ". " not in message
    assert "mp_physics=8" in message
    assert "qr_acr_qg_V4.dat" in message
    assert "woof fetch-tables" in message
    # A FileNotFoundError subclass, so every stage that already turns
    # OSError into a sentence prints it instead of a traceback.
    assert isinstance(raised.value, FileNotFoundError)


def test_a_fresh_wheel_runs_mp8_with_no_manual_fetch(tmp_path, monkeypatch,
                                                     capsys):
    """`mp_physics=8 (Thompson) needs qr_acr_qg_V4.dat, freezeH2O.dat,
    which are not staged at .../site-packages/woof_data/...`

    A clean ``pip install recast-woof`` refused every Thompson forecast until
    someone ran ``woof fetch-tables`` by hand (the image build and the
    2x MPAS box, 2026-10-16).  The run's own preflight now stages the
    set on first use, through the fetch-tables leg and its pins, into
    the user root; the advisory preflight still never downloads.
    """

    real_packaged = packaged_thompson_table_root()
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")
    _packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    fetched: list[str] = []

    def _pinned_source(root, asset):
        # Stands in for the network only; the bytes are still verified
        # against the pins before the atomic install.
        fetched.append(asset.filename)
        return table_assets.fetch_asset_from_dir(root, asset, real_packaged)

    monkeypatch.setattr(table_assets, "fetch_asset_from_sources",
                        _pinned_source)

    with pytest.raises(table_assets.MissingTableAssets):
        table_assets.require_thompson_tables(stage=False)
    assert fetched == []

    assert table_assets.require_thompson_tables() == user_root
    assert sorted(fetched) == sorted(table_assets.EXTERNALIZED_TABLE_FILENAMES)
    validate_table_assets(user_root)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "first use" in captured.err

    fetched.clear()
    assert table_assets.require_thompson_tables() == user_root
    assert fetched == []


def test_drift_stays_the_byte_validator_s_sentence_not_the_preflight_s(
        tmp_path, monkeypatch):
    """Negative control: the preflight answers absence, and only absence.

    A table that is PRESENT but wrong is a different failure with a
    better sentence available -- ``validate_table_assets`` can name the
    expected and actual byte counts.  The cheap gate must not shadow it
    with "not staged, run fetch-tables", which would send an operator
    with a drifted mirror to download a file they already have.
    """

    packaged = packaged_thompson_table_root()
    root = tmp_path / "drifted"
    root.mkdir()
    for asset in CLASSIC_TABLE_ASSETS:
        (root / asset.filename).write_bytes(b"drift")
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(root))
    assert table_assets.unstaged_table_assets() == []
    assert table_assets.require_thompson_tables() == root
    with pytest.raises(ValueError, match="has 5 bytes; expected"):
        validate_table_assets(root)

    # ... and a genuinely absent set is the preflight's to name.
    (root / CLASSIC_TABLE_ASSETS[0].filename).unlink()
    with pytest.raises(table_assets.MissingTableAssets):
        table_assets.require_thompson_tables()

    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(packaged))
    assert table_assets.unstaged_table_assets() == []


def test_gpuwm_check_warns_about_unstaged_tables_and_still_passes(
        tmp_path, monkeypatch, capsys):
    """Warn, never block: `woof check` is the memory preflight.

    Sizing a domain whose tables live elsewhere is legitimate, so this
    is one line and the exit code is untouched -- the run doors are
    where the same condition is refused.
    """

    from woof.core.preflight import _warn_unstaged_physics_tables

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(empty))

    class _Domain:
        def __init__(self, mp):
            self.run = type("R", (), {"mp_physics": mp})()

    exp = type("E", (), {"domains": (_Domain(8),)})()
    assert _warn_unstaged_physics_tables(exp) is None
    warned = capsys.readouterr().err
    assert warned.startswith("warning: ")
    assert "qr_acr_qg_V4.dat" in warned
    assert "woof fetch-tables" in warned

    # Negative control: a non-mp8 config says nothing at all.
    other = type("E", (), {"domains": (_Domain(10),)})()
    assert _warn_unstaged_physics_tables(other) is None
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# The staged root has to be complete for mp=28 too, and the two legs of
# the command are independent
# ---------------------------------------------------------------------------

def _packaged_ccn_asset():
    """The one aerosol asset, or a skip when this checkout lacks it."""

    from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

    return AEROSOL_TABLE_ASSETS[0]


def test_staging_carries_the_mp28_activation_table_into_the_staged_root(
        tmp_path, monkeypatch, capsys):
    """`MissingAerosolTableAsset: CCN_ACTIVATE.BIN ... was not found`

    Reproduced on the published 2.6.1 wheel: `woof fetch-tables` staged
    the four classic tables into ~/.woof/tables/thompson, printed "table
    root ... complete and byte-valid", exited 0 -- and the next
    mp_physics=28 forecast died at its first microphysics step, because
    the aerosol contract resolves CCN_ACTIVATE.BIN against the SAME root
    as the classic tables and the copy was still sitting in
    site-packages.  "Complete" has to mean complete for every scheme the
    root serves, not just for mp=8.
    """

    asset = _packaged_ccn_asset()
    real_packaged = packaged_thompson_table_root()
    if not (real_packaged / asset.filename).is_file():
        pytest.skip(f"packaged asset absent: {asset.filename}")
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    # A real wheel carries the activation table; the helper above only
    # lays down the classic set, so put it where a wheel has it.
    import shutil

    shutil.copyfile(real_packaged / asset.filename,
                    packaged / asset.filename)

    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    printed = capsys.readouterr().out
    assert asset.filename in printed
    assert "the mp=28 activation table" in printed

    staged = user_root / asset.filename
    assert staged.is_file(), "the staged root is still fatal for mp=28"
    assert staged.stat().st_size == asset.bytes

    # The check the first microphysics step makes, against the root a
    # run on this install actually resolves.
    from woof.core.thompson_aerosol_contract import (
        resolve_ccn_activation_path,
        validate_ccn_activation_asset,
    )

    assert resolve_ccn_activation_path(None, user_root) == staged
    assert validate_ccn_activation_asset(staged) == asset


def test_a_present_activation_table_is_left_alone(tmp_path, monkeypatch,
                                                  capsys):
    """Idempotent: the copy is a gap-filler, never an overwrite."""

    asset = _packaged_ccn_asset()
    real_packaged = packaged_thompson_table_root()
    if not (real_packaged / asset.filename).is_file():
        pytest.skip(f"packaged asset absent: {asset.filename}")
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    import shutil

    shutil.copyfile(real_packaged / asset.filename,
                    packaged / asset.filename)
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    first = (user_root / asset.filename).stat().st_mtime_ns
    capsys.readouterr()

    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    printed = capsys.readouterr().out
    assert "the mp=28 activation table" not in printed
    assert (user_root / asset.filename).stat().st_mtime_ns == first


def test_a_refused_wif_leg_still_stages_the_mandatory_tables(
        tmp_path, monkeypatch, capsys):
    """`woof fetch-tables --wif` used to stage NOTHING when --wif failed.

    Reproduced on the published 2.6.1 wheel against its own default URL:
    the WIF leg ran first, 404'd, returned 2, and the classic tables --
    which no flag makes optional and every forecast reads -- were never
    attempted.  The user asked for both and got neither.  The legs are
    independent; only --wif-only skips the mandatory one.
    """

    real_packaged = packaged_thompson_table_root()
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    wif_root = tmp_path / "wif"

    # --from carries the classic tables but not the WIF dataset, so the
    # optional leg refuses for a real reason while the mandatory one can
    # be satisfied.
    code = table_assets.fetch_tables_main(_args(
        from_dir=str(real_packaged), wif=True, wif_root=str(wif_root)))
    printed = capsys.readouterr().out

    assert code == 2, "a refused optional leg still fails the command"
    assert "REFUSED" in printed
    # ...and the mandatory leg ran anyway.
    validate_table_assets(user_root)
    assert table_assets.unstaged_table_assets() == []
    # The summary says which leg did what, so the exit code is not the
    # only thing the reader has.
    assert "classic coefficient tables: staged" in printed
    assert "aerosol climatology dataset: REFUSED" in printed


def test_wif_only_is_the_one_flag_that_skips_the_mandatory_leg(
        tmp_path, monkeypatch, capsys):
    """The operator saying so in as many words is still honoured."""

    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    wif_root = tmp_path / "wif"
    code = table_assets.fetch_tables_main(_args(
        from_dir=str(tmp_path / "empty"), wif=True, wif_only=True,
        wif_root=str(wif_root)))
    assert code == 2
    assert not user_root.exists(), "--wif-only touched the classic root"


def test_the_wif_dataset_resolves_from_its_fixed_public_source(
        monkeypatch):
    """A fixed dataset must not depend on the current binary release."""

    from woof import bridge_assets

    monkeypatch.delenv(table_assets.ASSET_URL_BASE_ENV, raising=False)
    monkeypatch.delenv("WOOF_BRIDGE_ASSET_URL_BASE", raising=False)

    class _Pins:
        release = "v9.9.9"

    monkeypatch.setattr(bridge_assets, "load_pins", lambda: _Pins())
    from woof.ingest.wif_dataset import WIF_DATASET_DOWNLOAD_BASE, WIF_DATASET_FILE
    assert table_assets.wif_asset_url_base() == WIF_DATASET_DOWNLOAD_BASE
    assert table_assets.wif_asset_source() == (
        WIF_DATASET_DOWNLOAD_BASE + "/" + WIF_DATASET_FILE + ".bz2", "bz2")


def test_a_source_install_without_release_pins_can_fetch_wif(
        monkeypatch):
    """Missing bridge-release metadata cannot strand fixed input data."""

    from woof import bridge_assets
    from woof.ingest.wif_dataset import WIF_DATASET_FILE

    monkeypatch.delenv(table_assets.ASSET_URL_BASE_ENV, raising=False)
    monkeypatch.delenv("WOOF_BRIDGE_ASSET_URL_BASE", raising=False)

    def _no_release():
        raise bridge_assets.BridgeAssetError("the packaged pins declare "
                                             "no release")

    monkeypatch.setattr(bridge_assets, "load_pins", _no_release)
    url, compression = table_assets.wif_asset_source()
    assert url.endswith(WIF_DATASET_FILE + ".bz2")
    assert compression == "bz2"


def test_the_wif_leg_stages_from_a_local_file_url_base(tmp_path,
                                                       monkeypatch, capsys):
    """The whole route, over a file:// base, with the real pin enforced."""

    from woof.ingest.wif_dataset import (
        WIF_DATASET_ASSET, WIF_DATASET_FILE)

    payload = tmp_path / "published"
    payload.mkdir()
    source = _real_wif_bytes()
    if source is None:
        pytest.skip("the pinned WIF dataset is not present on this host")
    import shutil

    shutil.copyfile(source, payload / WIF_DATASET_FILE)

    base = payload.resolve().as_uri()
    monkeypatch.setenv(table_assets.ASSET_URL_BASE_ENV, base)
    wif_root = tmp_path / "wif"
    assert table_assets.stage_wif_dataset(None, str(wif_root)) == 0
    landed = wif_root / WIF_DATASET_FILE
    assert landed.is_file()
    assert landed.stat().st_size == WIF_DATASET_ASSET.bytes
    assert "verified and installed" in capsys.readouterr().out


def _real_wif_bytes():
    """The pinned dataset wherever this host keeps it, else None."""

    from woof.ingest.wif_dataset import (
        WIF_DATASET_ASSET, WIF_DATASET_FILE, resolve_wif_data_root)

    candidates = [resolve_wif_data_root(None) / WIF_DATASET_FILE]
    for candidate in candidates:
        if (candidate.is_file()
                and candidate.stat().st_size == WIF_DATASET_ASSET.bytes):
            return candidate
    return None


# ---------------------------------------------------------------------------
# A damaged activation table is found, and repaired where this command owns
# the root; a stalled download ends in words
# ---------------------------------------------------------------------------

def _wheel_with_staged_user_root(tmp_path, monkeypatch):
    asset = _packaged_ccn_asset()
    real_packaged = packaged_thompson_table_root()
    if not (real_packaged / asset.filename).is_file():
        pytest.skip(f"packaged asset absent: {asset.filename}")
    if not all((real_packaged / name).is_file()
               for name in table_assets.EXTERNALIZED_TABLE_FILENAMES):
        pytest.skip("externalized asset bytes not present in this checkout")
    packaged, user_root = _wheel_shaped_install(tmp_path, monkeypatch)
    import shutil

    shutil.copyfile(real_packaged / asset.filename, packaged / asset.filename)
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(real_packaged))) == 0
    return asset, packaged, user_root


def test_an_empty_activation_table_in_the_staged_root_is_repaired(
        tmp_path, monkeypatch, capsys):
    """`has 0 bytes; expected 35288` used to be reported as verified.

    fetch-tables skipped any CCN_ACTIVATE.BIN it found, said "nothing to
    fetch", and the first mp=28 forecast refused the file at load; setup
    and reinstalling never touched the user copy.
    """

    from woof.core.thompson_aerosol_contract import (
        validate_ccn_activation_asset)

    asset, _packaged, user_root = _wheel_with_staged_user_root(
        tmp_path, monkeypatch)
    capsys.readouterr()
    (user_root / asset.filename).write_bytes(b"")

    assert table_assets.fetch_tables_main(_args()) == 0
    printed = capsys.readouterr().out
    assert "damaged" in printed and asset.filename in printed
    assert "nothing to fetch" not in printed
    assert validate_ccn_activation_asset(user_root / asset.filename) == asset


def test_an_empty_activation_table_in_a_named_root_is_refused_not_overwritten(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    asset = _packaged_ccn_asset()
    (root / asset.filename).write_bytes(b"")
    assert table_assets.fetch_tables_main(_args()) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and asset.filename in out
    assert (root / asset.filename).read_bytes() == b""


def test_doctor_does_not_verify_a_root_with_an_empty_activation_table(
        tmp_path, monkeypatch):
    from woof import doctor

    asset, _packaged, user_root = _wheel_with_staged_user_root(
        tmp_path, monkeypatch)
    assert doctor._thompson_tables_check().status == "verified"
    (user_root / asset.filename).write_bytes(b"")
    check = doctor._thompson_tables_check()
    assert check.status == "missing"
    assert asset.filename in check.detail
    assert check.action == "woof fetch-tables"
    (user_root / asset.filename).unlink()
    absent = doctor._thompson_tables_check()
    assert absent.status == "missing" and not absent.blocking
    assert "mp_physics=28" in absent.detail
    # Damaged or absent, only mp_physics=28 reads the file, so neither
    # fails the check for the Thompson runs that never open it.
    assert not check.blocking
    assert (check.blocking, check.severity) == (absent.blocking, absent.severity)
    assert "mp_physics=28" in check.detail and "mp_physics=28" in check.brief


def test_doctor_reads_the_activation_table_the_loader_would_read(
        tmp_path, monkeypatch):
    """The file override the mp=28 loader honours is the one checked."""

    from woof import doctor
    from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_PATH_ENV

    asset, packaged, user_root = _wheel_with_staged_user_root(
        tmp_path, monkeypatch)
    (user_root / asset.filename).write_bytes(b"")
    monkeypatch.setenv(AEROSOL_TABLE_PATH_ENV, str(packaged / asset.filename))
    assert doctor._thompson_tables_check().status == "verified"
    monkeypatch.setenv(AEROSOL_TABLE_PATH_ENV,
                       str(tmp_path / "elsewhere" / asset.filename))
    check = doctor._thompson_tables_check()
    assert check.status == "missing" and not check.blocking
    assert AEROSOL_TABLE_PATH_ENV in check.detail


def test_a_server_that_withholds_the_body_is_a_refusal_within_the_timeout(
        tmp_path, monkeypatch):
    """HTTP 200 and a length, then nothing: this used to wait forever.

    Since acb3e817fdf (bounded mesh fetch) a 200's Content-Length proves the
    object length, so the refusal now comes from the stall guard rather than
    the length probe. The stall window is shortened here so the bound is
    checked in seconds; production keeps its 30 s window.
    """

    import http.server
    import threading
    import time as clock

    release = threading.Event()

    class Withhold(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the handler's own spelling
            self.send_response(200)
            self.send_header("Content-Length", "8192")
            self.end_headers()
            self.wfile.flush()
            release.wait(30)

        def log_message(self, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Withhold)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    asset = _asset_for(b"\x00" * 8192)
    root = tmp_path / "root"
    root.mkdir()
    url = f"http://127.0.0.1:{server.server_address[1]}/{asset.filename}"
    monkeypatch.setenv("RUSTWX_DOWNLOAD_STALL_SECONDS", "1")
    started = clock.monotonic()
    try:
        with pytest.raises(table_assets.TableAssetError) as excinfo:
            table_assets.fetch_asset_from_url(root, asset, url)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
    # 1 s stall window, the client retries with backoff: about 7 s on a development machine.
    assert clock.monotonic() - started < 30
    message = str(excinfo.value)
    assert "stalled" in message, message
    assert not (root / asset.filename).exists()
    assert list(root.glob(".*fetch-partial*")) == []


def test_a_missing_file_on_the_server_is_not_retried(tmp_path, monkeypatch):
    calls = []
    real = table_assets._rw_asset

    def counting(kind, **kwargs):
        calls.append(kind)
        return real(kind, **kwargs)

    monkeypatch.setattr(table_assets, "_rw_asset", counting)
    asset = _asset_for(b"x" * 16)
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(table_assets.TableAssetError, match="nowhere.dat"):
        table_assets.fetch_asset_from_url(
            root, asset, (tmp_path / "nowhere.dat").as_uri())
    assert len(calls) == 1
