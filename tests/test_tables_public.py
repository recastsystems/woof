"""Downstream table inspection and repair use the loader's actual pins."""
import shutil

import pytest

from woof import tables
from woof.core.p3_tables import TABLE_1_2MOM_ASSET, packaged_p3_table_root
from woof.table_assets import fetch_tables_main


def test_status_partitions_missing_valid_and_same_size_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_TABLE_ROOT", str(tmp_path))
    required = tables.status(["p3"])
    assert required.needed == required.missing
    assert not required.present and not required.corrupt
    item = required.missing[0]
    item.path.parent.mkdir(parents=True)
    shutil.copyfile(packaged_p3_table_root() / TABLE_1_2MOM_ASSET.filename, item.path)
    valid = tables.status({"mp_physics": 50})
    assert valid.ready and valid.present == required.needed
    raw = bytearray(item.path.read_bytes())
    original = item.path.stat()
    raw[-50] ^= 1
    item.path.write_bytes(raw)
    import os
    os.utime(item.path, ns=(original.st_atime_ns, original.st_mtime_ns))
    damaged = tables.status(["p3"])
    assert damaged.corrupt == required.needed
    with pytest.raises(tables.CorruptTableAssets, match="woof fetch-tables --schemes p3"):
        tables.ensure(["p3"], fetch=False)


def test_missing_error_names_the_file_and_offline_command(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_TABLE_ROOT", str(tmp_path))
    with pytest.raises(tables.MissingTableAssets) as caught:
        tables.ensure(["noah"], fetch=False)
    assert "VEGPARM.TBL" in str(caught.value)
    assert "woof fetch-tables --schemes noah --from DIR" in str(caught.value)


def test_offline_repair_refuses_bad_source_and_keeps_existing_bytes(tmp_path, monkeypatch):
    from argparse import Namespace
    source = tmp_path / "source"
    source.mkdir()
    asset = TABLE_1_2MOM_ASSET
    shutil.copyfile(packaged_p3_table_root() / asset.filename, source / asset.filename)
    monkeypatch.setenv("WOOF_TABLE_ROOT", str(tmp_path / "tables"))
    args = Namespace(schemes=["p3"], from_dir=str(source), replace_damaged=False)
    assert fetch_tables_main(args) == 0
    item = tables.status(["p3"]).present[0]
    item.path.write_bytes(b"damaged destination")
    assert fetch_tables_main(args) == 2
    args.replace_damaged = True
    (source / asset.filename).write_bytes(b"bad source")
    assert fetch_tables_main(args) == 2
    assert item.path.read_bytes() == b"damaged destination"
    assert not list(item.path.parent.glob(".*fetch-partial*"))
    shutil.copyfile(packaged_p3_table_root() / asset.filename, source / asset.filename)
    assert fetch_tables_main(args) == 0
    assert tables.ensure(["p3"], fetch=False).ready


def test_generated_microphysics_needs_no_external_table_files():
    assert tables.status(["nssl", "morrison"]).needed == ()
    assert tables.ensure({"mp_physics": 10}, fetch=False).ready


def test_an_inactive_chemistry_flag_does_not_require_unused_aerosol_tables():
    report = tables.status({"mp_physics": 6, "ra_physics": 4, "aer_ra_feedback": 1})
    assert not any("aerosols-merra" in item.path.name for item in report.needed)


def test_loader_rejects_parseable_modified_noah_coefficients(tmp_path):
    from woof.core.noah import TBL_DIR, load_tables
    for name in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        shutil.copyfile(TBL_DIR / name, tmp_path / name)
    path = tmp_path / "GENPARM.TBL"
    raw = path.read_bytes()
    # A harmless ASCII header edit keeps the parser viable and tests the pin.
    path.write_bytes(raw.replace(b"General", b"general", 1))
    with pytest.raises(tables.CorruptTableAssets, match="GENPARM.TBL"):
        load_tables(tbl_dir=tmp_path)


def test_unknown_family_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown table family"):
        tables.status(["unknown-scheme"])


def test_load_cache_reuses_unchanged_bytes_but_detects_replacement(tmp_path, monkeypatch):
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor
    from woof import file_digest
    path = tmp_path / "lookup.dat"
    path.write_bytes(b"coefficients")
    item = tables.TableFile("fixture", path, path.stat().st_size,
                            hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []
    original = file_digest._hash
    def count(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(file_digest, "_hash", count)
    with ThreadPoolExecutor(4) as pool:
        assert all(pool.map(tables._valid_for_load, [item] * 8))
    assert len(calls) == 1
    old = path.stat()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"coefficientS")
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    replacement.replace(path)
    assert not tables._valid_for_load(item)
    assert len(calls) == 2


def test_public_inspection_bypasses_successful_loader_cache(tmp_path, monkeypatch):
    import hashlib
    path = tmp_path / "lookup.dat"
    path.write_bytes(b"coefficients")
    item = tables.TableFile("fixture", path, path.stat().st_size,
                            hashlib.sha256(path.read_bytes()).hexdigest())
    assert tables._valid_for_load(item)
    monkeypatch.setattr(tables, "_requests", lambda cfg: [("fixture", {})])
    monkeypatch.setattr(tables, "_files", lambda *args: [item])
    calls = []
    original = tables._valid
    def count(current):
        calls.append(current.path)
        return original(current)
    monkeypatch.setattr(tables, "_valid", count)
    assert tables.status(["fixture"]).ready
    assert tables.ensure(["fixture"], fetch=False).ready
    assert calls == [path, path]


def test_thompson_generation_staging_and_loading_share_the_default_root(tmp_path, monkeypatch):
    from pathlib import Path
    from woof.core.thompson_column import table_root, GENERATION_ASSETS
    from woof.config import RunConfig
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("WOOF_THOMPSON_GENERATION_TABLE_ROOT", raising=False)
    cfg = RunConfig(nx=12, ny=12, nz=16, dx=3000., dy=3000.,
                    ztop=15000., dt=1., run_seconds=1., moist=True,
                    mp_physics=8, thompson_version="wrf_44")
    expected = tmp_path / ".woof/tables/thompson-wrf44"
    assert table_root(cfg) == expected
    staged = tables._files("thompson44", {})
    loaded = tables._files("thompson44", cfg)
    for name, _, _ in GENERATION_ASSETS["wrf_44"]:
        assert expected / name in [item.path for item in staged]
        assert expected / name in [item.path for item in loaded]


def test_fork_public_ensure_preserves_the_acquisition_error(tmp_path, monkeypatch):
    from woof import thompson_fork_assets
    item = tables.TableFile("thompson-fork", tmp_path / "lookup.dat", 1, "")
    report = tables.TableStatus((item,), (), (item,), ())
    monkeypatch.setattr(tables, "_status", lambda *a, **k: report)
    error = tables.TableAssetError("offline https://example.invalid/lookup.dat; woof fetch-tables --schemes thompson-fork --from DIR")
    def refuse(*args, **kwargs):
        raise error
    monkeypatch.setattr(thompson_fork_assets, "ensure_thompson_fork_tables", refuse)
    with pytest.raises(tables.TableAssetError) as caught:
        tables.ensure(["thompson-fork"])
    assert caught.value is error


def test_interrupted_fragment_cleanup_preserves_live_writers(tmp_path):
    import os
    import subprocess
    import sys
    from woof import table_assets
    from woof.core.thompson_contract import TableAsset
    pid = subprocess.Popen([sys.executable, "-c", "pass"])
    pid.wait(timeout=10)
    prefix = f".lookup.dat.fetch-partial-{pid.pid}-" + "a" * 32
    stale = tmp_path / (prefix + ".part-" + str(pid.pid) + "-1")
    child_live = tmp_path / (prefix + ".part-" + str(os.getpid()) + "-2")
    parent_live = tmp_path / (f".lookup.dat.fetch-partial-{os.getpid()}-" + "b" * 32)
    for path in (stale, child_live, parent_live):
        path.write_bytes(b"unpublished")
    table_assets._cleanup_interrupted(tmp_path, TableAsset("lookup.dat", 0, ""))
    assert not stale.exists()
    assert child_live.read_bytes() == parent_live.read_bytes() == b"unpublished"


@pytest.mark.parametrize("error_type", [tables.MissingTableAssets, tables.TableAssetError])
def test_cli_table_failures_are_one_named_refusal(error_type, monkeypatch, capsys):
    from woof import cli
    def refuse(args):
        raise error_type("p3 needs lookup.dat; run woof fetch-tables --schemes p3 --from DIR")
    monkeypatch.setattr(cli, "_dispatch", refuse)
    assert cli.main(["fetch-tables"]) == 2
    error = capsys.readouterr().err
    assert "p3 needs lookup.dat" in error
    assert "woof fetch-tables --schemes p3 --from DIR" in error
    assert "Traceback" not in error
