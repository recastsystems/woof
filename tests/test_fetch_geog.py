"""``woof fetch-geog``: staging, verification, and refusal contract.

The WPS_GEOG tree is user-staged data the whole static path depends on;
fetch-geog is its installer.  These tests bind the pin table to the
static builder's dataset list, exercise the resumable download and
skip-complete logic against a fake transport, prove the extraction
safety and index-validation bars on tiny synthetic tarballs, and pin
the doctor/wizard remedy text to the command.  A live network smoke is
marked ``network`` and additionally gated on ``WOOF_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from woof import geog_assets
from woof.domain_wizard import GEOG_DATASETS

#: The pins that stage the WRF builder's nine default directories, in table
#: order: eight, because the two default soil-category directories
#: (soilgrids/texture_top, texture_bot) are children of one archive.
WRF_PINS = tuple(a.dataset for a in geog_assets.GEOG_ARCHIVES
                 if a.dataset in geog_assets.archive_names_for_paths(GEOG_DATASETS))
from woof.geog_assets import (
    GEOG_FETCH_MANIFEST_NAME,
    GEOG_FETCH_MANIFEST_SCHEMA,
    GeogArchive,
    GeogFetchError,
    archive_url,
    default_geog_root,
    fetch_geog,
    parse_datasets,
    resolve_source,
    validate_dataset_dir,
)

_INDEX_TEXT = """\
type = continuous
projection = regular_ll
dx = 0.5
dy = 0.5
known_x = 1.0
known_y = 1.0
known_lat = -89.75
known_lon = -179.75
wordsize = 2
tile_x = 4
tile_y = 4
tile_z = 1
"""


# ---------------------------------------------------------------------------
# Synthetic archive construction + fake transport
# ---------------------------------------------------------------------------

def _build_archive(datasets: tuple[str, ...], *, mode: str = "w:bz2",
                   with_index: bool = True,
                   evil_member: str | None = None,
                   prefix: str = "") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as tar:
        for bare in datasets:
            name = f"{prefix}{bare}"
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
            tile = b"\x00\x01" * 16
            data = tarfile.TarInfo(f"{name}/00001-00004.00001-00004")
            data.size = len(tile)
            tar.addfile(data, io.BytesIO(tile))
            if with_index:
                payload = _INDEX_TEXT.encode()
                index = tarfile.TarInfo(f"{name}/index")
                index.size = len(payload)
                tar.addfile(index, io.BytesIO(payload))
        if evil_member is not None:
            info = tarfile.TarInfo(evil_member)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"evil"))
    return buffer.getvalue()


def _add_sidecars(datasets: tuple[str, ...], *, mode: str = "w:bz2",
                  prefix: str = "") -> bytes:
    """An archive with AppleDouble junk beside real members (as NCAR's
    mandatory bundle genuinely carries)."""
    base = _build_archive(datasets, mode=mode, prefix=prefix)
    buffer = io.BytesIO(base)
    out = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="r:*") as src, \
            tarfile.open(fileobj=out, mode=mode) as dst:
        for member in src:
            payload = src.extractfile(member) if member.isfile() else None
            dst.addfile(member, payload)
        for name in datasets:
            junk = tarfile.TarInfo(f"{prefix}._{name}")
            junk.size = 3
            dst.addfile(junk, io.BytesIO(b"AD\x00"))
            inner = tarfile.TarInfo(f"{prefix}{name}/._index")
            inner.size = 3
            dst.addfile(inner, io.BytesIO(b"AD\x00"))
    return out.getvalue()


def _pin(dataset: str, payload: bytes,
         filename: str | None = None) -> GeogArchive:
    return GeogArchive(dataset, filename or f"{dataset}.tar.bz2",
                       len(payload), hashlib.sha256(payload).hexdigest(),
                       4096)


class _FakeResponse:
    """A stub that answers like a real origin, headers included.

    A ``206`` without ``Content-Range`` is not a thing HTTP allows, and
    a stub that omits it lets a resume gate pass vacuously -- the
    transport is supposed to check that the server is sending the bytes
    it asked for, and it cannot check a header the stub never sends.
    """

    def __init__(self, payload: bytes, status: int = 200,
                 headers: dict[str, str] | None = None):
        self._data = io.BytesIO(payload)
        self.status = status
        self.headers = dict(headers or {})

    def read(self, n: int = -1) -> bytes:
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_transport(files: dict[str, bytes], *, honor_range: bool = True,
                    etag: str = '"pinned-object-v1"'):
    """urlopen_fn serving ``files`` by basename, recording every request."""

    calls: list = []

    def urlopen_fn(request):
        calls.append(request)
        name = request.full_url.rsplit("/", 1)[1]
        payload = files[name]
        header = request.headers.get("Range")
        if header and honor_range:
            offset = int(header.split("=", 1)[1].rstrip("-"))
            return _FakeResponse(payload[offset:], status=206, headers={
                "ETag": etag,
                "Content-Range":
                    f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
            })
        return _FakeResponse(payload, headers={"ETag": etag})

    urlopen_fn.calls = calls
    return urlopen_fn


def _refusing_transport():
    def urlopen_fn(request):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"network touched: {request.full_url}")
    return urlopen_fn


# ---------------------------------------------------------------------------
# Pin table and argument parsing
# ---------------------------------------------------------------------------

def test_pin_table_binds_the_static_builders_dataset_list():
    # The WRF static builder's datasets must every one be fetchable.  The
    # pin table is allowed to be wider than that list -- the MPAS static
    # builder reads Noah-MP soil products the WRF one never opens -- so the
    # bar is containment, not equality.  Equality would make adding a
    # dataset for one builder look like a defect in the other.
    fetchable = geog_assets.geog_datasets()
    # A directory is fetchable when a pin stages it, at the top or as a
    # declared child of a container archive (soilgrids/texture_top).
    archive_names = geog_assets.archive_names_for_paths(GEOG_DATASETS)
    assert archive_names <= set(fetchable)
    assert len(set(fetchable)) == len(fetchable), "duplicate pin"
    # The wizard's list IS the builder's default selection.
    from woof.static.build import _DEFAULT_GEOG_DIRS
    assert set(GEOG_DATASETS) == set(_DEFAULT_GEOG_DIRS.values())
    # Which door needs a pin is a COLUMN of the table, not a branch in the
    # code that reads it.  The WRF-scoped column is exactly the pins that
    # stage the nine the WRF static builder opens: a pin drifting into that
    # column would have doctor demand a dataset that builder never opens.
    assert (geog_assets.datasets_required_by(geog_assets.GEOG_CONSUMER_WRF)
            == WRF_PINS)
    # Licence: nothing on a door's required column, and nothing in the
    # default fetch, carries a licence hold (the project ruling, 2026-10-14/2026-10-21).
    for archive in geog_assets.GEOG_ARCHIVES:
        if archive.licence_hold:
            assert not archive.required_by and not archive.fetch_by_default
            assert "hf" not in archive.available_sources
    for archive in geog_assets.GEOG_ARCHIVES:
        assert archive.required_by or archive.optional_for, f"{archive.dataset} serves no door"
        unknown = set(archive.required_by + archive.optional_for) - set(geog_assets.GEOG_CONSUMERS)
        assert not unknown, f"{archive.dataset}: unknown consumer {unknown}"


def test_lake_depth_is_explicitly_fetchable_without_changing_mandatory_wrf_data():
    archive = geog_assets.archive_for("lake_depth")
    assert archive.filename == "lake_depth.tar.bz2"
    assert archive.archive_bytes == 2425674
    assert archive.archive_sha256 == "7f016173f4999e67d9757bddb9e80f1b1f3794bfe820fd78b215eb599d59fd76"
    assert archive.extracted_bytes == 1884949370
    assert archive.required_by == ()
    assert archive.optional_for == (geog_assets.GEOG_CONSUMER_WRF,)
    assert archive.available_sources == ("ncar",)
    assert not archive.in_mandatory_bundle
    assert parse_datasets("lake_depth") == ("lake_depth",)
    assert "lake_depth" not in parse_datasets("wrf")
    assert archive_url(archive.filename, "ncar") == (
        "https://www2.mmm.ucar.edu/wrf/src/wps_files/lake_depth.tar.bz2")


def test_uncopied_optional_archive_uses_upstream_and_records_actual_source(tmp_path, monkeypatch):
    from dataclasses import replace
    payload = _build_archive(("lake_depth",))
    pin = replace(_pin("lake_depth", payload), required_by=(),
                  optional_for=(geog_assets.GEOG_CONSUMER_WRF,), available_sources=("ncar",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    transport = _fake_transport({pin.filename: payload})
    assert fetch_geog(root=tmp_path, datasets=("lake_depth",), source="hf",
                      urlopen_fn=transport, progress=lambda message: None) == 1
    assert transport.calls[0].full_url.startswith(geog_assets.NCAR_BASE_URL)
    manifest = json.loads((tmp_path / GEOG_FETCH_MANIFEST_NAME).read_text())
    record = manifest["archives"][pin.filename]
    assert record["source"] == "ncar"
    assert record["url"].startswith(geog_assets.NCAR_BASE_URL)


@pytest.mark.parametrize("dataset,size,digest,unpacked", [
    ("bnu_soiltype_top", 8198836,
     "7a7eb86d585c3dc6b32297f1ea4622d929aacadc70f16e14e0193a06f233038a",
     933120270),
    ("bnu_soiltype_bot", 8078617,
     "49306298749e3ed2a6172dfe7af0023a5be0e9f5f5230a2cfd6421ebce3e81b1",
     933120273),
])
def test_bnu_soil_pins_are_optional_ncar_data(dataset, size, digest, unpacked):
    archive = geog_assets.archive_for(dataset)
    assert archive.filename == f"{dataset}.tar.bz2"
    assert archive.archive_bytes == size
    assert archive.archive_sha256 == digest
    assert archive.extracted_bytes == unpacked
    assert archive.required_by == ()
    assert archive.optional_for == (geog_assets.GEOG_CONSUMER_WRF,)
    assert archive.available_sources == ("ncar",)
    assert not archive.in_mandatory_bundle
    assert parse_datasets(dataset) == (dataset,)
    assert dataset not in parse_datasets("wrf")
    assert dataset not in geog_assets.MANDATORY_BUNDLE_DATASETS
    assert archive_url(archive.filename, "ncar") == (
        f"https://www2.mmm.ucar.edu/wrf/src/wps_files/{dataset}.tar.bz2")


def test_thirty_second_lai_pin_is_optional_ncar_data():
    # The HRRR static source's off-grid recipe (modis_lai) reads it; the
    # WRF door's mandatory nine do not change.
    archive = geog_assets.archive_for("lai_modis_30s")
    assert archive.filename == "lai_modis_30s.tar.bz2"
    assert archive.archive_bytes == 749633975
    assert archive.archive_sha256 == (
        "b8188f06717767d5ed8898e5afb3a9b8b5ad46b888a530177f161ef3623acf75")
    assert archive.required_by == ()
    assert archive.optional_for == (geog_assets.GEOG_CONSUMER_WRF,)
    assert archive.available_sources == ("ncar",)
    assert parse_datasets("lai_modis_30s") == ("lai_modis_30s",)
    assert "lai_modis_30s" not in parse_datasets("wrf")
    assert "lai_modis_30s" not in geog_assets.MANDATORY_BUNDLE_DATASETS
    assert archive_url(archive.filename, "ncar") == (
        "https://www2.mmm.ucar.edu/wrf/src/wps_files/lai_modis_30s.tar.bz2")


def test_explicit_bnu_soil_fetch_uses_ncar_and_records_both_archives(
        tmp_path, monkeypatch):
    from dataclasses import replace

    datasets = ("bnu_soiltype_top", "bnu_soiltype_bot")
    files = {}
    pins = []
    for dataset in datasets:
        payload = _build_archive((dataset,))
        archive = geog_assets.archive_for(dataset)
        pins.append(replace(archive, archive_bytes=len(payload),
                            archive_sha256=hashlib.sha256(payload).hexdigest(),
                            extracted_bytes=4096))
        files[archive.filename] = payload
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", tuple(pins))
    transport = _fake_transport(files)
    assert fetch_geog(root=tmp_path, datasets=datasets, source="ncar",
                      urlopen_fn=transport, progress=lambda message: None) == 2
    assert len(transport.calls) == 2
    manifest = json.loads((tmp_path / GEOG_FETCH_MANIFEST_NAME).read_text())
    for archive in pins:
        record = manifest["archives"][archive.filename]
        assert record["source"] == "ncar"
        assert record["url"] == archive_url(archive.filename, "ncar")
        assert (tmp_path / archive.dataset / "index").is_file()
    assert all(request.full_url.startswith(geog_assets.NCAR_BASE_URL)
               for request in transport.calls)


def test_the_mesh_column_carries_every_dataset_the_mesh_door_opens():
    # rw_mpas_static's required list is dataset DIRECTORIES; the pin table
    # is ARCHIVES, and one archive carries seven directories.  The two are
    # bound through the pin table's own child declarations, so a dataset
    # added to the engine's list without a mesh-scoped pin behind it is a
    # tree doctor calls complete and the door then refuses.  The scope is
    # every dataset a mesh-door option reads: the native default and
    # --legacy-static, whose with-lakes land use the static builder
    # woof hex 0.4.0 ships also requires (2.9.1: scoped WRF-only, `woof
    # fetch-geog --datasets mesh` staged an archive every hex point mesh
    # refused).
    from woof.rustwx_static import (
        LEGACY_GEOG_DATASETS, REQUIRED_GEOG_DATASETS, fetchable_for)

    mesh_pins = set(geog_assets.datasets_required_by(
        geog_assets.GEOG_CONSUMER_MESH))
    read = tuple(dict.fromkeys(REQUIRED_GEOG_DATASETS + LEGACY_GEOG_DATASETS))
    assert set(fetchable_for(read)) == mesh_pins
    assert "soilgrids" in mesh_pins
    assert "modis_landuse_20class_30s_with_lakes" in mesh_pins


def test_the_soil_archive_is_required_by_both_doors():
    # 2026-10-21: the WRF builder's default soil categories are the
    # SoilGrids tiles inside this archive (the FAO-based soiltype_*_30s are
    # held out for their licence), so both doors require it.
    archive = geog_assets.archive_for("soilgrids")
    assert archive.required_by == (geog_assets.GEOG_CONSUMER_WRF,
                                   geog_assets.GEOG_CONSUMER_MESH)
    assert "soilgrids" in geog_assets.datasets_required_by(
        geog_assets.GEOG_CONSUMER_WRF)


def test_the_noahmp_soil_container_is_pinned_and_declares_its_children():
    # rw_mpas_static requires soilcomp and four soil texture layers.  They
    # arrive in one upstream archive as child directories, each with its
    # own WPS index, so the pin has to name the children or a staged tree
    # cannot be validated.
    archive = geog_assets.archive_for("soilgrids")
    assert archive.filename == "soilgrids.tar.bz2"
    assert set(archive.index_subdirs) == {
        "soilcomp", "texture_top", "texture_bot",
        "texture_layer1", "texture_layer2",
        "texture_layer3", "texture_layer4",
    }
    # The five rw_mpas_static reads must each be a declared child.
    for child in ("soilcomp", "texture_layer1", "texture_layer2",
                  "texture_layer3", "texture_layer4"):
        assert child in archive.index_subdirs


def test_the_reader_finds_every_mesh_dataset_where_the_fetcher_stages_it(
        tmp_path):
    """The fetcher's layout and the door's reader, bound through one table.

    The fetcher validates a container archive child by child at
    ``root/<archive>/<child>/index``; the reader behind doctor's mesh row
    and `woof mesh`'s refusal used to look at ``root/<child>/index``
    alone.  Both passed their own tests and every install that followed
    doctor's remedy was told it was still five datasets short.  This
    stages a tree the fetcher's own bar accepts and asks the reader.
    """

    from woof import rustwx_static

    root = tmp_path / "WPS_GEOG"
    for archive in geog_assets.GEOG_ARCHIVES:
        if geog_assets.GEOG_CONSUMER_MESH not in archive.required_by:
            continue
        for child in archive.index_subdirs or ("",):
            directory = root / archive.dataset / child
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "index").write_text(_INDEX_TEXT)
        ok, detail = validate_dataset_dir(root, archive.dataset,
                                          check_receipt=False)
        assert ok, detail
    assert rustwx_static.missing_geog_datasets(root) == []
    for name in rustwx_static.REQUIRED_GEOG_DATASETS:
        assert rustwx_static.geog_dataset_dir(root, name) is not None, name


def test_the_mandatory_bundle_claims_only_what_was_walked():
    # --bundle serves the nine the 2026-07-29 walk found.  A pin added
    # later must not silently widen that claim, or --bundle extracts
    # nothing for it and reports success.
    assert geog_assets.MANDATORY_BUNDLE_DATASETS == frozenset((
        "topo_gmted2010_30s", "modis_landuse_20class_30s_with_lakes",
        "soiltype_top_30s", "soiltype_bot_30s", "greenfrac_fpar_modis",
        "lai_modis_10m", "albedo_modis", "maxsnowalb_modis", "soiltemp_1deg"))
    assert "soilgrids" not in geog_assets.MANDATORY_BUNDLE_DATASETS


def test_single_dataset_pins_declare_no_children():
    for archive in geog_assets.GEOG_ARCHIVES:
        if archive.dataset in GEOG_DATASETS:
            assert archive.index_subdirs == (), archive.dataset


def test_pins_are_real_digests_and_positive_sizes():
    for archive in geog_assets.GEOG_ARCHIVES:
        if archive.producer:
            # Built, not downloaded: no tarball to pin; the size quoted is
            # the network read and the producer must exist.
            assert archive.producer in geog_assets.GEOG_PRODUCERS
            assert archive.filename == "" and archive.archive_sha256 == ""
            assert archive.archive_bytes > 0 and archive.extracted_bytes > 0
            continue
        assert len(archive.archive_sha256) == 64, archive.dataset
        int(archive.archive_sha256, 16)
        assert archive.archive_bytes > 0
        assert archive.extracted_bytes > archive.archive_bytes
        assert archive.filename == f"{archive.dataset}.tar.bz2"
    assert len(geog_assets.MANDATORY_BUNDLE_SHA256) == 64
    int(geog_assets.MANDATORY_BUNDLE_SHA256, 16)


def test_parse_datasets_all_and_subset_and_unknown():
    assert parse_datasets("all") == geog_assets.geog_datasets()
    # canonical order regardless of request order, duplicates collapsed
    picked = parse_datasets("soiltemp_1deg,topo_gmted2010_30s,soiltemp_1deg")
    assert picked == ("topo_gmted2010_30s", "soiltemp_1deg")
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_datasets("topo_gmted2010_30s,not_a_dataset")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_datasets(" , ")


def test_the_default_fetch_stages_the_soil_archive_too():
    # The ruling made the ~13 GB soil archive a sanctioned default
    # download for the mesh path.  A bare `woof fetch-geog` -- whose
    # --datasets default is what this parses -- must therefore include it,
    # or `woof mesh` refuses a tree the installer just called done.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    geog_assets.register_cli(sub)
    args = parser.parse_args(["fetch-geog"])
    assert parse_datasets(args.datasets) == tuple(
        a.dataset for a in geog_assets.GEOG_ARCHIVES if a.fetch_by_default)
    assert "soilgrids" in parse_datasets(args.datasets)


def test_a_consumer_name_selects_that_doors_datasets():
    # The subset spelling for an operator who knows they only want WRF:
    # a consumer name off the table's own scoping column, not a per-door
    # flag.  `--datasets wrf` is the WRF-only escape from the 13 GB.
    assert parse_datasets("wrf") == WRF_PINS
    assert "soilgrids" in parse_datasets("wrf")
    assert "soiltype_top_30s" not in parse_datasets("wrf")
    assert parse_datasets("mesh") == geog_assets.datasets_required_by(
        geog_assets.GEOG_CONSUMER_MESH)
    # and it composes with explicit names and the other door, in canonical
    # order (the mesh door's native static reads two land-use archives the
    # WRF builder does not, so wrf plus the soil archive is no longer all)
    assert parse_datasets("wrf,mesh") == tuple(
        archive.dataset for archive in geog_assets.GEOG_ARCHIVES
        if archive.required_by and archive.fetch_by_default)
    assert parse_datasets("wrf,soilgrids") == WRF_PINS
    explicit = ("wrf,mesh,lake_depth,orogwd3_10m,bnu_soiltype_top,"
                "bnu_soiltype_bot,orogwd_10m,varsso_10m")
    assert parse_datasets(explicit) == tuple(
        a.dataset for a in geog_assets.GEOG_ARCHIVES if a.fetch_by_default)


@pytest.mark.parametrize("dataset,size,digest,unpacked,children", [
    ("orogwd_10m", 3332672,
     "0e8c292c901e5148cb80849dc154d19cc885393658c5725741da597859777e5c",
     46658851, 10),
    ("orogwd3_10m", 14957055,
     "0ffb24427670e4114a8a64dc7106c260152e35ad0f3176e1da4149ed9eb10c84",
     93319204, 20),
    ("varsso_10m", 783440,
     "a2420ea9698a30afdb46a82a0c13961ca17603fb11a4e657c973cf2dff95c727",
     6221332, 0),
])
def test_orographic_drag_pins_are_ncar_data_in_the_default_fetch(
        dataset, size, digest, unpacked, children):
    # WOOF Global's default suite (gwd_opt 3) reads orogwd_10m and
    # orogwd3_10m; without these rows a bare `woof fetch-geog` left a
    # fresh machine's default Global run refusing for want of them.
    archive = geog_assets.archive_for(dataset)
    assert archive.filename == f"{dataset}.tar.bz2"
    assert (archive.archive_bytes, archive.archive_sha256,
            archive.extracted_bytes) == (size, digest, unpacked)
    assert len(archive.index_subdirs) == children
    assert archive.required_by == ()
    assert archive.optional_for == (geog_assets.GEOG_CONSUMER_WRF,)
    assert archive.available_sources == ("ncar",)
    assert not archive.in_mandatory_bundle
    assert dataset in parse_datasets("all")
    assert dataset not in parse_datasets("wrf")


def test_the_orographic_children_are_the_static_builders_leaves():
    # The child lists are the leaves woof/static/orographic.py opens
    # under each dataset, so validation looks where the reader reads.
    from woof.static import orographic

    leaves: dict[str, set[str]] = {}
    for row in orographic.OROGRAPHIC_ROWS.values():
        top, _, leaf = row.rel_path["default"].partition("/")
        leaves.setdefault(top, set()).add(leaf)
    assert set(leaves) == {"orogwd_10m", "orogwd3_10m", "varsso_10m"}
    for dataset, found in leaves.items():
        expected = set(geog_assets.archive_for(dataset).index_subdirs) or {""}
        assert found == expected, dataset


def test_resolve_source_defaults_and_bundle_rules():
    assert resolve_source(None, False) == "ncar"
    assert resolve_source("ncar", False) == "ncar"
    assert resolve_source(None, True) == "ncar"  # bundle implies ncar
    with pytest.raises(ValueError, match="NCAR only"):
        resolve_source("hf", True)
    with pytest.raises(ValueError, match="unknown --source"):
        resolve_source("ftp", False)


def test_archive_url_per_source_and_env_override(monkeypatch):
    monkeypatch.delenv(geog_assets.GEOG_URL_BASE_ENV, raising=False)
    assert archive_url("x.tar.bz2", "hf") == (
        f"{geog_assets.HF_MIRROR_BASE_URL}/x.tar.bz2")
    assert archive_url("x.tar.bz2", "ncar") == (
        f"{geog_assets.NCAR_BASE_URL}/x.tar.bz2")
    monkeypatch.setenv(geog_assets.GEOG_URL_BASE_ENV,
                       "https://mirror.example/geog/")
    assert archive_url("x.tar.bz2", "hf") == (
        "https://mirror.example/geog/x.tar.bz2")
    assert archive_url("x.tar.bz2", "ncar") == (
        "https://mirror.example/geog/x.tar.bz2")


def test_default_geog_root_is_the_case_data_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path))
    assert default_geog_root() == tmp_path / "WPS_GEOG"


# ---------------------------------------------------------------------------
# Dataset directory validation (the doctor bar, plus a parsing index)
# ---------------------------------------------------------------------------

def test_validate_dataset_dir_bars(tmp_path):
    ok, detail = validate_dataset_dir(tmp_path, "absent_ds")
    assert not ok and "does not exist" in detail

    (tmp_path / "noindex_ds").mkdir()
    ok, detail = validate_dataset_dir(tmp_path, "noindex_ds")
    assert not ok and "index" in detail

    hollow = tmp_path / "hollow_ds"
    hollow.mkdir()
    (hollow / "index").write_text("")  # vacuous: no keys at all
    ok, detail = validate_dataset_dir(tmp_path, "hollow_ds")
    assert not ok and "required WPS index key" in detail

    good = tmp_path / "good_ds"
    good.mkdir()
    (good / "index").write_text(_INDEX_TEXT)
    ok, detail = validate_dataset_dir(tmp_path, "good_ds")
    assert ok, detail


def test_validate_dataset_dir_checks_every_declared_child(tmp_path, monkeypatch):
    # A container archive's index lives one level down, once per child.
    # Every declared child has to pass: a tree missing one is a finding,
    # not a pass earned by its siblings.
    monkeypatch.setattr(
        geog_assets, "GEOG_ARCHIVES",
        (GeogArchive("box_ds", "box_ds.tar.bz2", 10, "0" * 64, 20,
                     ("first", "second")),))

    box = tmp_path / "box_ds"
    (box / "first").mkdir(parents=True)
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert not ok and "lacks its WPS `index` file" in detail

    (box / "first" / "index").write_text(_INDEX_TEXT)
    (box / "second").mkdir()
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert not ok and "second" in detail, detail

    (box / "second" / "index").write_text(_INDEX_TEXT)
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert ok, detail

    # A top-level index does not substitute for the children's.
    (box / "second" / "index").unlink()
    (box / "index").write_text(_INDEX_TEXT)
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert not ok and "second" in detail, detail


def _staged_tree(root: Path, dataset: str, projection: str | None) -> Path:
    """Stage one dataset directory whose index declares ``projection``.

    ``projection=None`` writes no projection row at all, which is the
    WPS default (regular_ll) and must stay readable.
    """
    directory = root / dataset
    directory.mkdir(parents=True)
    lines = [line for line in _INDEX_TEXT.splitlines()
             if not line.startswith("projection")]
    if projection is not None:
        lines.append(f"projection = {projection}")
    (directory / "index").write_text("\n".join(lines) + "\n")
    (directory / "00001-00004.00001-00004").write_bytes(b"\x00\x01" * 16)
    return directory


def test_validate_dataset_dir_names_an_index_projection_it_cannot_read(
        tmp_path):
    """The inventory names a projected tree, before anything downloads.

    Real WPS_GEOG ships projected tile sets, and such a tree parses
    perfectly while being unreadable: the reader resolves a source cell
    by dividing a longitude difference by the index dx.  Every door that
    consults this inventory (the listing, the pre-fetch skip decision,
    post-stage validation) therefore names it here rather than leaving
    it to surface at dataset construction inside a run.
    """
    _staged_tree(tmp_path, "projected_ds", "albers_nad83")
    ok, detail = validate_dataset_dir(tmp_path, "projected_ds",
                                      check_receipt=False)
    assert not ok, detail
    assert "albers_nad83" in detail                  # what was declared
    assert "equirectangular" in detail               # the assumption
    assert "wrong source cells" in detail            # the breakage
    assert "regular_ll" in detail                    # the way out


@pytest.mark.parametrize("projection", ["regular_ll", None])
def test_validate_dataset_dir_passes_equirectangular_trees(
        tmp_path, projection):
    """NEGATIVE CONTROL: the new bar refuses only what it names.

    The staged trees in use declare regular_ll, and an index with no
    projection row defaults to it, so both still validate and both
    still construct.
    """
    directory = _staged_tree(tmp_path, "plain_ds", projection)
    ok, detail = validate_dataset_dir(tmp_path, "plain_ds",
                                      check_receipt=False)
    assert ok, detail

    from woof.static.geog import GeogDataset
    assert GeogDataset(directory).index.projection == "regular_ll"


# ---------------------------------------------------------------------------
# The engine: skip, download, verify, extract, manifest
# ---------------------------------------------------------------------------

def test_fetch_stages_verifies_and_writes_manifest(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    transport = _fake_transport({"alpha_ds.tar.bz2": payload})
    root = tmp_path / "WPS_GEOG"

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None, urlopen_fn=transport)

    assert staged == 1
    ok, detail = validate_dataset_dir(root, "alpha_ds")
    assert ok, detail
    manifest = json.loads(
        (root / GEOG_FETCH_MANIFEST_NAME).read_text())
    assert manifest["schema"] == GEOG_FETCH_MANIFEST_SCHEMA
    entry = manifest["archives"]["alpha_ds.tar.bz2"]
    assert entry["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["pinned"] is True
    assert entry["source"] == "hf"
    assert entry["datasets"]["alpha_ds"]["files"] == 2
    # default: the verified archive is removed after extraction
    assert not (root / geog_assets.ARCHIVE_SUBDIR
                / "alpha_ds.tar.bz2").exists()
    # no extraction droppings
    assert not list(root.glob(f"{geog_assets.ARCHIVE_SUBDIR}-*"))
    assert not list((root / geog_assets.ARCHIVE_SUBDIR).glob("*.tmp"))


def test_fetch_prints_the_total_before_the_first_byte_moves(
        tmp_path, monkeypatch):
    """The size total is announced up front, not discovered per dataset.

    ``woof setup --with-geog`` promises "the size is printed before
    anything downloads"; the engine both entry points share must keep
    that promise for ``woof fetch-geog`` too -- the first field run
    watched 16 GB arrive with the only sizes printed as each dataset
    landed.  The total covers exactly the datasets that still need
    staging, not ones already valid on disk.
    """
    alpha = _build_archive(("alpha_ds",))
    beta = _build_archive(("beta_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", alpha), _pin("beta_ds", beta)))
    root = tmp_path / "WPS_GEOG"
    staged_dir = root / "alpha_ds"          # alpha is already valid,
    staged_dir.mkdir(parents=True)          # so only beta counts
    (staged_dir / "index").write_text(_INDEX_TEXT)

    events: list[str] = []
    inner = _fake_transport({"beta_ds.tar.bz2": beta})

    def transport(request):
        events.append("NETWORK")
        return inner(request)

    staged = fetch_geog(root=root, datasets=("alpha_ds", "beta_ds"), source="hf",
               progress=events.append, urlopen_fn=transport)

    totals = [line for line in events
              if isinstance(line, str) and "need staging" in line]
    assert len(totals) == 1, events
    total = totals[0]
    # Only beta's pin is in the total, and both numbers are stated.
    assert "1 of 2" in total
    assert f"{len(beta):,} B" in total          # download size
    assert "~4.0 KiB" in total                  # extracted_bytes pin (4096)
    # Printed before the first byte moved.
    assert events.index(total) < events.index("NETWORK")
    assert staged == 1


@pytest.mark.parametrize("honor_range", [True, False], ids=["resume", "restart"])
def test_fetch_resumes_a_partial_download_with_a_range_request(
        tmp_path, monkeypatch, honor_range):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    transport = _fake_transport({"alpha_ds.tar.bz2": payload}, honor_range=honor_range)
    root = tmp_path / "WPS_GEOG"
    partial = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:100] if honor_range else b"garbage-prefix")

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None, urlopen_fn=transport)

    assert staged == 1
    (request,) = transport.calls
    assert request.headers.get("Range") == f"bytes={100 if honor_range else len(b'garbage-prefix')}-"
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok


def test_fetch_reuses_a_complete_verified_archive(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    complete = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    complete.parent.mkdir(parents=True)
    complete.write_bytes(payload)

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None,
                        urlopen_fn=_refusing_transport())
    assert staged == 1
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok


def test_sha_mismatch_refuses_and_quarantines(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    wrong = payload + b"tail"
    pin = _pin("alpha_ds", payload)
    pin = GeogArchive(pin.dataset, pin.filename, len(wrong),
                      pin.archive_sha256, pin.extracted_bytes)
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    transport = _fake_transport({"alpha_ds.tar.bz2": wrong})
    root = tmp_path / "WPS_GEOG"

    with pytest.raises(GeogFetchError, match="does not match the pin"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None, urlopen_fn=transport)
    assert not (root / "alpha_ds").exists()
    rejected = list((root / geog_assets.ARCHIVE_SUBDIR).glob("*.rejected-*"))
    assert len(rejected) == 1


def test_upstream_drift_is_ncar_only_and_band_limited(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    pin = GeogArchive("alpha_ds", "alpha_ds.tar.bz2", len(payload),
                      "0" * 64, 4096)  # pin that can never match
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    root = tmp_path / "WPS_GEOG"

    # hf: drift is corruption by definition, flag or no flag
    with pytest.raises(GeogFetchError, match="corrupted transfer"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   allow_drift=True, progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))

    # ncar without the flag: refused with the drift story
    with pytest.raises(GeogFetchError, match="allow-upstream-drift"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))

    # ncar with the flag, size inside the band: accepted, marked unpinned
    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                        allow_drift=True, progress=lambda *_: None,
                        urlopen_fn=_fake_transport(
                            {"alpha_ds.tar.bz2": payload}))
    assert staged == 1
    manifest = json.loads((root / GEOG_FETCH_MANIFEST_NAME).read_text())
    assert manifest["archives"]["alpha_ds.tar.bz2"]["pinned"] is False


def test_upstream_drift_outside_the_size_band_refuses(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    # pin claims the archive should be far larger than what arrives:
    # with drift allowed the exact-size bar is off, so the verify
    # step's sanity band is what must catch the runt
    pin = GeogArchive("alpha_ds", "alpha_ds.tar.bz2", len(payload) * 10,
                      "0" * 64, 4096)
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="outside the sanity band"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                   allow_drift=True, progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))
    assert not (root / "alpha_ds").exists()


def test_keep_archives_retains_the_verified_tarball(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
               keep_archives=True, progress=lambda *_: None,
               urlopen_fn=_fake_transport({"alpha_ds.tar.bz2": payload}))
    kept = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    assert kept.read_bytes() == payload


# ---------------------------------------------------------------------------
# Extraction safety and validation bars
# ---------------------------------------------------------------------------

def test_archive_without_an_index_is_refused(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",), with_index=False)
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="failed validation"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))
    assert not (root / "alpha_ds").exists()


def test_traversal_member_is_refused(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",),
                             evil_member="alpha_ds/../../evil")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="unsafe member"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))
    assert not (tmp_path / "evil").exists()
    assert not (root / "alpha_ds").exists()


def test_symlink_member_is_refused(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        link = tarfile.TarInfo("alpha_ds/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/hosts"
        tar.addfile(link)
    payload = buffer.getvalue()
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="non-regular member"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))


def test_invalid_existing_dataset_is_quarantined_not_overwritten(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    broken = root / "alpha_ds"
    broken.mkdir(parents=True)
    (broken / "leftover").write_text("partial extraction debris")

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None,
                        urlopen_fn=_fake_transport(
                            {"alpha_ds.tar.bz2": payload}))
    assert staged == 1
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok
    quarantined = list(root.glob("alpha_ds.rejected-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "leftover").is_file()


# ---------------------------------------------------------------------------
# Bundle mode
# ---------------------------------------------------------------------------

def test_bundle_extracts_only_needed_datasets_from_one_archive(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds", "beta_ds", "extra_ds"),
                             mode="w:gz")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"unused-a"),
                         _pin("beta_ds", b"unused-b")))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_BYTES", len(payload))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset({"alpha_ds", "beta_ds"}))
    root = tmp_path / "WPS_GEOG"
    # beta_ds is already staged; only alpha_ds should be extracted
    staged_dir = root / "beta_ds"
    staged_dir.mkdir(parents=True)
    (staged_dir / "index").write_text(_INDEX_TEXT)

    transport = _fake_transport(
        {geog_assets.MANDATORY_BUNDLE_FILENAME: payload})
    staged = fetch_geog(root=root, datasets=("alpha_ds", "beta_ds"),
                        source="ncar", bundle=True,
                        progress=lambda *_: None, urlopen_fn=transport)
    assert staged == 1
    assert (root / "alpha_ds" / "index").is_file()
    assert not (root / "extra_ds").exists()
    (request,) = transport.calls
    assert request.full_url.endswith(geog_assets.MANDATORY_BUNDLE_FILENAME)


def test_bundle_wps_geog_prefix_layout_is_stripped(tmp_path, monkeypatch):
    """NCAR's mandatory bundle nests datasets under one WPS_GEOG/ dir
    (verified by walking the real archive); extraction must land them
    at root/<dataset> exactly like the per-dataset tarballs."""
    payload = _build_archive(("alpha_ds",), mode="w:gz",
                             prefix="WPS_GEOG/")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"unused"),))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_BYTES", len(payload))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset({"alpha_ds"}))
    root = tmp_path / "WPS_GEOG"
    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                        bundle=True, progress=lambda *_: None,
                        urlopen_fn=_fake_transport(
                            {geog_assets.MANDATORY_BUNDLE_FILENAME:
                             payload}))
    assert staged == 1
    assert (root / "alpha_ds" / "index").is_file()
    assert not (root / "WPS_GEOG").exists()


def test_appledouble_sidecars_are_never_extracted(tmp_path, monkeypatch):
    """NCAR's bundle carries macOS `._*` sidecars beside every dataset
    (verified by walking the real archive); neither route may land
    them in the staged tree."""
    payload = _add_sidecars(("alpha_ds",), mode="w:gz",
                            prefix="WPS_GEOG/")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"unused"),))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_BYTES", len(payload))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset({"alpha_ds"}))
    root = tmp_path / "WPS_GEOG"
    fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
               bundle=True, progress=lambda *_: None,
               urlopen_fn=_fake_transport(
                   {geog_assets.MANDATORY_BUNDLE_FILENAME: payload}))
    assert (root / "alpha_ds" / "index").is_file()
    assert not (root / "alpha_ds" / "._index").exists()
    assert not list(root.rglob("._*"))


def test_bundle_refuses_datasets_it_does_not_contain(tmp_path, monkeypatch):
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"x"),))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset())
    with pytest.raises(ValueError, match="does not contain"):
        fetch_geog(root=tmp_path / "WPS_GEOG", datasets=("alpha_ds",),
                   source="ncar", bundle=True, progress=lambda *_: None,
                   urlopen_fn=_refusing_transport())


# ---------------------------------------------------------------------------
# A failure on this computer is not a download failure, and a resume is
# promised only when a partial file can actually be resumed
# ---------------------------------------------------------------------------

_DEEP_WINDOWS_PATH = "C:\\" + "d" * 269          # 272 characters


def _download(dest: Path, payload: bytes, transport) -> None:
    geog_assets.download_archive(
        geog_assets.archive_url(dest.name, "hf"), dest,
        expected_bytes=len(payload), progress=lambda *_: None,
        label="alpha_ds", urlopen_fn=transport)


def test_a_resume_record_windows_refuses_is_reported_as_a_path_problem(
        tmp_path, monkeypatch):
    """The failure the deep install folder produced, with its real errno.

    Windows answered ENOENT for the 272-character staging path of the
    resume record.  That used to read "download failed from
    https://huggingface.co/...; the partial file is kept and a re-run
    resumes" -- no byte had been written, the mirror was fine, and a re-run
    failed identically.
    """

    payload = _build_archive(("alpha_ds",))
    dest = tmp_path / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    dest.parent.mkdir()

    def windows_refuses(path, text, *, tag="publish"):
        raise FileNotFoundError(2, "No such file or directory",
                                _DEEP_WINDOWS_PATH)

    monkeypatch.setattr(geog_assets.fetch_guard, "atomic_write_text",
                        windows_refuses)
    monkeypatch.setattr(geog_assets.fetch_guard, "windows_path_limit",
                        lambda: 259)

    with pytest.raises(GeogFetchError) as caught:
        _download(dest, payload, _fake_transport({dest.name: payload}))

    message = str(caught.value)
    assert "download failed" not in message
    assert "resumes" not in message
    assert "272 characters" in message
    assert "259" in message
    assert "13 characters shorter" in message
    assert "not a download failure" in message
    assert "nothing was downloaded" in message
    assert message.index("272 characters") < message.index(_DEEP_WINDOWS_PATH)
    assert not dest.exists()


def test_a_local_failure_off_windows_names_the_path_without_blaming_length(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    dest = tmp_path / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    dest.parent.mkdir()

    def disk_full(path, text, *, tag="publish"):
        raise OSError(28, "No space left on device", str(path))

    monkeypatch.setattr(geog_assets.fetch_guard, "atomic_write_text",
                        disk_full)
    monkeypatch.setattr(geog_assets.fetch_guard, "windows_path_limit",
                        lambda: None)

    with pytest.raises(GeogFetchError) as caught:
        _download(dest, payload, _fake_transport({dest.name: payload}))

    message = str(caught.value)
    sidecar = geog_assets._resume_sidecar(dest)
    assert "download failed" not in message
    assert "No space left on device" in message
    assert f"{len(str(sidecar))} characters" in message
    assert "Windows" not in message
    assert "resumes" not in message


def test_a_network_failure_before_any_byte_promises_no_resume(tmp_path):
    payload = _build_archive(("alpha_ds",))
    dest = tmp_path / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    dest.parent.mkdir()

    def unreachable(request):
        raise geog_assets.URLError("connection refused")

    with pytest.raises(GeogFetchError) as caught:
        _download(dest, payload, unreachable)

    message = str(caught.value)
    assert "download failed from https://huggingface.co/" in message
    assert "resumes" not in message
    assert "nothing was downloaded" in message


class _DroppedConnection(_FakeResponse):
    """Sends the first ``keep`` bytes, then the connection resets."""

    def __init__(self, payload: bytes, keep: int, **kwargs):
        super().__init__(payload[:keep], **kwargs)
        self._sent = False

    def read(self, n: int = -1) -> bytes:
        if self._sent:
            raise ConnectionResetError(104, "Connection reset by peer")
        self._sent = True
        return super().read(n)


def test_a_dropped_connection_keeps_a_partial_that_a_rerun_resumes(tmp_path):
    payload = _build_archive(("alpha_ds",))
    dest = tmp_path / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    dest.parent.mkdir()

    def dropped(request):
        return _DroppedConnection(payload, 100,
                                  headers={"ETag": '"pinned-object-v1"'})

    with pytest.raises(GeogFetchError) as caught:
        _download(dest, payload, dropped)

    message = str(caught.value)
    assert "download failed from https://huggingface.co/" in message
    assert "the partial file (100 B) is kept and a re-run resumes" in message
    assert dest.stat().st_size == 100

    transport = _fake_transport({dest.name: payload})
    _download(dest, payload, transport)
    (request,) = transport.calls
    assert request.headers.get("Range") == "bytes=100-"
    assert dest.read_bytes() == payload


def test_the_resume_promise_follows_what_the_next_run_will_do(tmp_path):
    """The note mirrors download_archive's own resume rule, case by case."""

    dest = tmp_path / "alpha_ds.tar.bz2"
    sidecar = geog_assets._resume_sidecar(dest)
    note = geog_assets._partial_note

    assert note(dest, sidecar, strict_size=True) == (
        "nothing was downloaded, so a re-run starts from the beginning")
    dest.write_bytes(b"")
    assert "nothing was downloaded" in note(dest, sidecar, strict_size=True)
    dest.write_bytes(b"A" * 100)
    # Pinned: the exact size and SHA-256 pin catches any mixture, so an
    # unrecorded partial is still resumed.
    assert "(100 B) is kept and a re-run resumes" in note(
        dest, sidecar, strict_size=True)
    # No pin and no record: the next run sets it aside and restarts.
    unpinned = note(dest, sidecar, strict_size=False)
    assert "resumes" not in unpinned
    assert "downloads it from the start" in unpinned
    sidecar.write_text("{}", encoding="utf-8")
    assert "(100 B) is kept and a re-run resumes" in note(
        dest, sidecar, strict_size=False)


def test_an_extraction_windows_refuses_is_a_path_problem_and_keeps_the_archive(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    monkeypatch.setattr(geog_assets.fetch_guard, "windows_path_limit",
                        lambda: 259)
    root = tmp_path / "WPS_GEOG"
    real_open = Path.open

    def refusing_open(self, mode="r", *args, **kwargs):
        if "w" in mode and self.name == "00001-00004.00001-00004":
            raise FileNotFoundError(2, "No such file or directory",
                                    _DEEP_WINDOWS_PATH)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refusing_open)
    with pytest.raises(GeogFetchError) as caught:
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   keep_archives=False, progress=lambda *_: None,
                   urlopen_fn=_fake_transport({"alpha_ds.tar.bz2": payload}))
    monkeypatch.setattr(Path, "open", real_open)

    message = str(caught.value)
    archive = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    assert "272 characters" in message
    assert "13 characters shorter" in message
    assert "not a download failure" in message
    assert "verified archive is kept" in message
    assert archive.read_bytes() == payload
    assert not (root / "alpha_ds").exists()


def test_geography_staging_adds_no_more_than_the_longest_published_name():
    """Every staging name is compact, so a folder deep enough to hold the
    published geography is deep enough to stage it.

    Measured at the deepest WRF dataset: the published tile
    ``modis_landuse_20class_30s_with_lakes/00001-01200.00001-01200`` is 61
    characters below the geography folder.  Staging used to reach 110 (the
    resume record's staging copy) and 101 (the extraction folder).
    """

    root = Path("G")
    deepest_tile = (root / "modis_landuse_20class_30s_with_lakes"
                    / "00001-01200.00001-01200")
    archive = (root / geog_assets.ARCHIVE_SUBDIR
               / "modis_landuse_20class_30s_with_lakes.tar.bz2")
    sidecar = geog_assets._resume_sidecar(archive)
    sidecar_staging = geog_assets.fetch_guard._staging_path(sidecar, "geog")
    extraction = geog_assets._extraction_staging(root)
    staged_tile = extraction / deepest_tile.relative_to(root)

    def below(path: Path) -> int:
        return len(str(path)) - len(str(root))

    assert below(deepest_tile) == 61
    assert below(sidecar) == 75
    assert below(sidecar_staging) <= below(sidecar)
    assert below(staged_tile) <= below(deepest_tile) + 24


def _max_path_applies() -> bool:
    """Read independently of the code under test, which also decides it."""

    if os.name != "nt":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem"
                            ) as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] != 1
    except OSError:
        return True


@pytest.mark.skipif(not _max_path_applies(),
                    reason="MAX_PATH applies only to Windows without long "
                           "paths")
def test_a_deep_windows_geography_folder_on_the_real_file_system(
        tmp_path, monkeypatch):
    """The real Windows path, not a mock of it.

    A geography folder of 170 characters holds the published tree (231 at
    its deepest tile) and used to fail anyway on the resume record's
    280-character staging copy.  At 200 characters the resume record
    itself (275) cannot exist, and the refusal says so in characters.
    """

    payload = _build_archive(("modis_landuse_20class_30s_with_lakes",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (_pin(
        "modis_landuse_20class_30s_with_lakes", payload),))

    def folder(length: int) -> Path:
        path = tmp_path / str(length)
        assert len(str(path)) <= length - 2
        while len(str(path)) < length:
            remaining = length - len(str(path))
            name = min(60, remaining - 1)
            if remaining - 1 - name == 1:
                name -= 1
            path = path / ("p" * name)
        assert len(str(path)) == length
        return path

    fits = folder(170)
    fetch_geog(root=fits, datasets=("modis_landuse_20class_30s_with_lakes",),
               source="hf", progress=lambda *_: None,
               urlopen_fn=_fake_transport(
                   {"modis_landuse_20class_30s_with_lakes.tar.bz2": payload}))
    ok, detail = validate_dataset_dir(
        fits, "modis_landuse_20class_30s_with_lakes")
    assert ok, detail

    deep = folder(200)
    with pytest.raises(GeogFetchError) as caught:
        fetch_geog(root=deep,
                   datasets=("modis_landuse_20class_30s_with_lakes",),
                   source="hf", progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"modis_landuse_20class_30s_with_lakes.tar.bz2":
                        payload}))
    message = str(caught.value)
    assert "download failed" not in message
    assert "resumes" not in message
    assert "characters" in message and "259" in message


# ---------------------------------------------------------------------------
# Remedy text: doctor and wizard point at this command
# ---------------------------------------------------------------------------

def test_doctor_remedy_is_the_fetch_geog_command():
    from woof.doctor import GEOG_HINT
    assert "woof fetch-geog" in GEOG_HINT
    # the accurate size warning survives rewording
    assert "GB" in GEOG_HINT


def test_wizard_geog_help_names_the_downloader(capsys):
    from woof.domain_wizard import _print_geog_help
    _print_geog_help()
    out = capsys.readouterr().out
    assert "woof fetch-geog" in out
    for dataset in GEOG_DATASETS:
        assert dataset in out


# ---------------------------------------------------------------------------
# CLI wiring (through the real dispatcher)
# ---------------------------------------------------------------------------

def test_cli_list_mode_runs_offline_through_dispatch(tmp_path, capsys):
    from woof.cli import main
    code = main(["fetch-geog", "--list", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "fetch-geog: total" in out
    for dataset in WRF_PINS:
        assert dataset in out


def _stage_every_index(root: Path, datasets: tuple[str, ...]) -> None:
    """A root on which every requested dataset validates as staged."""
    for dataset in datasets:
        subdirs = geog_assets.archive_for(dataset).index_subdirs
        for directory in ([root / dataset] if not subdirs
                          else [root / dataset / name for name in subdirs]):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "index").write_text(_INDEX_TEXT)


def test_list_on_a_fully_staged_root_prints_no_download(tmp_path, capsys):
    """hrrr-full-09 (b): a fully staged root listed every dataset "staged"
    and then ended "total 1.21 GiB download", a download that a fetch of
    that root would never make."""
    import re
    from woof.cli import main
    datasets = parse_datasets("wrf")
    _stage_every_index(tmp_path, datasets)
    assert main(["fetch-geog", "--datasets", "wrf", "--list",
                 "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "  needed  " not in out
    assert out.count("  staged  ") == len(datasets)
    assert f"all {len(datasets)} requested dataset(s) are staged" in out
    assert "nothing to download" in out
    downloads = [float(size) for size in
                 re.findall(r"([0-9.]+) GiB download", out)]
    assert not any(downloads), out


def test_list_totals_only_the_datasets_still_needed(tmp_path, capsys):
    """The total is the sum of the "needed" rows, the same bill fetch prints."""
    from woof.cli import main
    datasets = parse_datasets("wrf")
    # The largest archive is left unstaged, so the total is a figure the
    # two-decimal GiB format cannot round to zero.
    needed = max(datasets,
                 key=lambda name: geog_assets.archive_for(name).archive_bytes)
    _stage_every_index(tmp_path, tuple(name for name in datasets
                                       if name != needed))
    assert main(["fetch-geog", "--datasets", "wrf", "--list",
                 "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    archive = geog_assets.archive_for(needed)
    assert archive.archive_bytes >= 0.01 * 1024 ** 3
    assert f"  needed  {needed}:" in out
    assert (f"fetch-geog: total {archive.archive_bytes / 1024 ** 3:.2f} GiB "
            "download") in out
    assert f"for the 1 of {len(datasets)} dataset(s) still needed" in out


@pytest.mark.parametrize("extra", [
    ["--datasets", "mesh"],
    ["--datasets", "mesh", "--source", "ncar"],
    ["--source", "ncar", "--bundle"],
    ["--datasets", "all", "--keep-archives"],
    ["--static-source", "hrrr-conus-v4"],
])
def test_explain_prints_the_plan_and_touches_nothing(
        tmp_path, monkeypatch, capsys, extra):
    """Hex global run, 2026-10-16: `woof fetch-geog --explain` started a
    13 GB download.  --explain is print-only on every path: exit 0, the plan
    on stdout, no network and no writes."""
    import socket

    from woof.cli import main
    from woof.static import external_source

    def refuse(*_args, **_kwargs):
        raise AssertionError("--explain reached the network or the fetcher")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(geog_assets, "fetch_geog", refuse)
    monkeypatch.setattr(geog_assets, "download_archive", refuse)
    monkeypatch.setattr(geog_assets, "_rw_asset", refuse)
    monkeypatch.setattr(external_source, "fetch_static_source", refuse)
    root = tmp_path / "geog"
    assert main(["fetch-geog", "--explain", "--root", str(root), *extra]) == 0
    out = capsys.readouterr().out
    assert not root.exists()
    if "--static-source" in extra:
        assert "would stage static source hrrr-conus-v4" in out
    else:
        assert "nothing was downloaded or written" in out
    if extra[:2] == ["--datasets", "mesh"]:
        # The two native land-use pins the hex static opens, with the
        # host the fetch will really use (they are not on the mirror).
        for name in ("modis_landuse_20class_30s", "landuse_30s"):
            assert f"  needed  {name}: " in out
            assert f"wps_files/{name}.tar.bz2" in out


def test_cli_refuses_unknown_dataset_as_a_usage_error(tmp_path, capsys):
    from woof.cli import main
    code = main(["fetch-geog", "--list", "--root", str(tmp_path),
                 "--datasets", "not_a_dataset"])
    assert code == 2
    assert "unknown dataset" in capsys.readouterr().err


def test_cli_refuses_bundle_from_the_mirror(tmp_path, capsys):
    from woof.cli import main
    code = main(["fetch-geog", "--list", "--root", str(tmp_path),
                 "--source", "hf", "--bundle"])
    assert code == 2
    assert "NCAR only" in capsys.readouterr().err


def test_fetch_geog_main_defaults_root_to_case_data_layout(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path))
    from woof.cli import main
    code = main(["fetch-geog", "--list"])
    out = capsys.readouterr().out
    assert code == 0
    assert str(tmp_path / "WPS_GEOG") in out


# ---------------------------------------------------------------------------
# Live network smoke (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
def test_live_upstream_archives_match_the_pins():
    """HEAD every pinned archive on NCAR; sizes must equal the pins."""
    from urllib.request import Request, urlopen

    for archive in geog_assets.GEOG_ARCHIVES:
        url = archive_url(archive.filename, "ncar")
        request = Request(url, method="HEAD",
                          headers={"User-Agent": "gpuwm-tests/1"})
        with urlopen(request, timeout=60) as response:
            assert response.status == 200, url
            length = int(response.headers["Content-Length"])
        assert length == archive.archive_bytes, (
            f"{archive.filename}: NCAR now serves {length} B, pin is "
            f"{archive.archive_bytes} B -- upstream drift; refresh the "
            "pins and the mirror together")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
def test_live_mirror_serves_the_pinned_bytes_sizes():
    """HEAD the mirror; skip (not fail) while the repo is unpublished."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    for archive in geog_assets.GEOG_ARCHIVES:
        if "hf" not in archive.available_sources:
            continue
        url = archive_url(archive.filename, "hf")
        request = Request(url, method="HEAD",
                          headers={"User-Agent": "gpuwm-tests/1"})
        try:
            with urlopen(request, timeout=60) as response:
                assert response.status == 200, url
                length = int(response.headers["Content-Length"])
        except HTTPError as error:
            # HF answers 401 (not 404) for a repo that does not exist
            # or is private to anonymous callers
            if error.code in (401, 403, 404):
                pytest.skip(f"mirror not published yet "
                            f"(HTTP {error.code}): {url}")
            raise
        assert length == archive.archive_bytes, url


def _publish_tool():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "publish_geog_mirror.py"
    spec = importlib.util.spec_from_file_location("publish_geog_mirror", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mirror_publisher_knows_every_pinned_archive():
    """Every archive the pin table verifies has a provenance row, and the
    mirror README renders with all of them.

    The breakage this prevents: soilgrids was pinned on 2026-08-23 and the
    publisher, written for nine tarballs, had no row for it -- so the
    mirror lagged the pin table by a whole archive and every mesh-door
    fetch 404'd on the default source until 2026-09-01.  A pinned archive
    the publisher cannot render must fail here, not on the next mirror
    snapshot.
    """
    tool = _publish_tool()
    # The mirror carries downloaded tarballs without a licence hold; a
    # produced dataset has no tarball, and a held pin (FAO-UNESCO soil,
    # ISLSCP-I soiltemp_1deg) is never republished.
    mirrored = [a for a in geog_assets.GEOG_ARCHIVES
                if a.producer is None and not a.licence_hold]
    assert tool.GEOG_ARCHIVES == tuple(mirrored)
    missing = [a.dataset for a in mirrored if a.dataset not in tool._PROVENANCE]
    assert missing == [], f"publisher has no provenance row for {missing}"
    stale = sorted(set(tool._PROVENANCE) - {a.dataset for a in geog_assets.GEOG_ARCHIVES})
    assert stale == [], f"publisher carries provenance for unpinned {stale}"
    readme = tool._readme_text()
    for archive in mirrored:
        assert f"`{archive.filename}` | {archive.archive_bytes:,} | `{archive.archive_sha256}`" in readme
        assert f"| `{archive.dataset}` |" in readme
    for name in ("soiltype_top_30s", "soiltype_bot_30s", "soiltemp_1deg",
                 "soiltemp_era5_1deg"):
        assert f"`{name}.tar.bz2`" not in readme


def test_every_dataset_the_native_mpas_static_path_reads_is_pinned():
    """The native static path is table work on both sides: the crate's
    option table names a WPS_GEOG directory for every native selector, and
    each must be a pin `woof fetch-geog` can stage through rw_fetch.

    The breakage this prevents: a native selector (or the native default:
    ``landuse_30s`` for the GWD statistics, ``modis_landuse_20class_30s``
    for MODIFIED_IGBP_MODIS_NOAH) naming a directory no pin stages, so a
    native static on a clean install stops at "WPS_GEOG index ... is
    unreadable" with no fetch that could fix it.
    """
    import re
    crate = Path(__file__).resolve().parents[1] / "tools/rustwx/crates/rw-mpas/src"
    builder = (crate / "static_builder.rs").read_text(encoding="utf-8")
    table = builder[builder.index("pub const MPAS_STATIC_SOURCE_TABLE"):]
    table = table[:table.index("];")]
    native = set(re.findall(r'\(\s*"[a-z-]+"\s*,\s*"[A-Za-z0-9_]+"\s*,\s*"([a-z0-9_]+)"\s*\)', table))
    defaults = builder[builder.index("pub const GEOG_DATASET_TABLE"):]
    defaults = defaults[:defaults.index("];")]
    native |= set(re.findall(r'\(\s*"[a-z_0-9]+"\s*,\s*"([a-z0-9_]+)"\s*\)', defaults))
    gwd = (crate / "native_gwd.rs").read_text(encoding="utf-8")
    native |= set(re.findall(r'\(\s*"(?:GTOPO30|GMTED2010)"\s*,\s*"([a-z0-9_]+)"', gwd))
    native |= set(re.findall(r'root\.join\("([a-z0-9_]+)"\)', gwd))
    assert {"landuse_30s", "modis_landuse_20class_30s", "topo_30s",
            "topo_gmted2010_30s"} <= native, native
    pinned = set(geog_assets.geog_datasets())
    for archive in geog_assets.GEOG_ARCHIVES:
        pinned.update(archive.index_subdirs)
    assert sorted(native - pinned) == []
    for name in ("landuse_30s", "modis_landuse_20class_30s"):
        archive = geog_assets.archive_for(name)
        assert geog_assets.GEOG_CONSUMER_MESH in archive.required_by, name
        assert archive.available_sources == ("ncar",) and not archive.in_mandatory_bundle
    for name in ("topo_30s", "modis_landuse_20class_15s", "greenfrac",
                 "albedo_ncep", "maxsnowalb"):
        archive = geog_assets.archive_for(name)
        assert archive.required_by == () and geog_assets.GEOG_CONSUMER_MESH in archive.optional_for
        assert not archive.fetch_by_default, f"{name} would join the default fetch bill"
        assert archive.download_source == "ncar" and archive.available_sources == ("ncar",)
        assert not archive.in_mandatory_bundle


def test_fetch_skips_already_valid_datasets_without_network(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    staged_dir = root / "alpha_ds"
    staged_dir.mkdir(parents=True)
    (staged_dir / "index").write_text(_INDEX_TEXT)

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None,
                        urlopen_fn=_refusing_transport())
    assert staged == 0
