"""Dust pins and the optional row-driven Rust contract, CPU only."""
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof import geog_assets as geog
from woof.static import rust_bridge
from woof.static.extra_fields import build_extra_fields

DUST = ("erod", "clayfrac_5m", "sandfrac_5m")
ROOT = Path(__file__).resolve().parents[1]


def test_dust_pins_and_default_scope():
    from woof.bridges import BRIDGE_ABI_MARKERS, BRIDGE_OPTIONAL_ABI_MARKERS
    assert BRIDGE_ABI_MARKERS["static_fields"] == rust_bridge.ABI_MARKER
    assert BRIDGE_OPTIONAL_ABI_MARKERS["static_fields"]["extra_continuous"] == rust_bridge.EXTRA_ABI_MARKER
    assert geog.datasets_required_by("chem-dust") == DUST
    assert geog.parse_datasets("chem-dust") == DUST
    from woof.domain_wizard import GEOG_DATASETS
    # The builder's nine directories, owned by eight pins (the two default
    # soil-category directories are children of the soilgrids archive).
    wrf_pins = geog.archive_names_for_paths(GEOG_DATASETS)
    assert set(geog.datasets_required_by("wrf")) == wrf_pins
    assert len(GEOG_DATASETS) == 9
    assert set(geog.parse_datasets("default")) == (
        wrf_pins | {"soilgrids", "lake_depth",
                              "bnu_soiltype_top", "bnu_soiltype_bot",
                              # the mesh door's native static land use (2.9.0)
                              "modis_landuse_20class_30s", "landuse_30s",
                              # the packaged HRRR recipe's gwd_opt 3 terrain drag,
                              # fetched by default since 959e1f06016
                              # (development branch)
                              "orogwd3_10m",
                              # the orographic drag statistics a default
                              # WOOF Global run reads (ddc882f020)
                              "orogwd_10m", "varsso_10m"})
    assert set(geog.parse_datasets("default")).isdisjoint(DUST)
    for name in DUST:
        pin = geog.archive_for(name)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.archive_sha256)
        assert pin.archive_bytes > 0 and pin.extracted_bytes > pin.archive_bytes
        assert pin.required_by == ("chem-dust",)
        assert not pin.in_mandatory_bundle and not pin.fetch_by_default
        assert pin.download_source == "ncar"
        assert pin.pin_date == "2026-09-30" and pin.pin_host == "www2.mmm.ucar.edu"
        assert not pin.index_subdirs


@pytest.mark.parametrize("library", [SimpleNamespace(),
    SimpleNamespace(gpuwm_static_extra_continuous_v1=lambda: 1)])
def test_old_library_refuses_only_extra_fields(monkeypatch, library):
    monkeypatch.setattr(rust_bridge, "_LIBRARY", library)
    assert rust_bridge.load() is library
    with pytest.raises(rust_bridge.StaticBridgeError, match="cannot build extra static fields.*rebuild"):
        rust_bridge.build_extra_fields(1, [])


def test_missing_dataset_names_remedy(tmp_path):
    with pytest.raises(FileNotFoundError, match="erod.*fetch-geog --datasets chem-dust"):
        build_extra_fields(None, tmp_path, [{"dataset_path": "erod"}])
    with pytest.raises(ValueError, match="under geog_root"):
        build_extra_fields(None, tmp_path, [{"dataset_path": "../outside"}])
    assert build_extra_fields(None, tmp_path, []) == {}


def test_upstream_only_row_uses_ncar_with_default_hf(tmp_path, monkeypatch, capsys):
    from dataclasses import replace
    from test_fetch_geog import _build_archive, _pin, _fake_transport
    payload = _build_archive(("arbitrary",))
    pin = replace(_pin("arbitrary", payload), download_source="ncar",
                  available_sources=("ncar",))
    monkeypatch.setattr(geog, "GEOG_ARCHIVES", (pin,))
    transport = _fake_transport({pin.filename: payload})
    geog.fetch_geog(root=tmp_path, datasets=(pin.dataset,), source="hf",
                    urlopen_fn=transport)
    assert transport.calls[0].full_url == geog.archive_url(pin.filename, "ncar")
    assert "unavailable on hf; using ncar" in capsys.readouterr().out
    receipt = json.loads((tmp_path / geog.GEOG_FETCH_MANIFEST_NAME).read_text())
    assert receipt["archives"][pin.filename]["source"] == "ncar"


def test_optional_consumer_doctor(tmp_path):
    from woof.doctor import _geog_tree_checks
    checks = _geog_tree_checks(tmp_path, consumers=("chem-dust",))
    assert len(checks) == 1 and checks[0].status == "missing"
    assert checks[0].action == "woof fetch-geog --datasets chem-dust"
    for dataset in DUST:
        (tmp_path / dataset).mkdir()
        (tmp_path / dataset / "index").write_text("type=continuous\n")
    checks = _geog_tree_checks(tmp_path, consumers=("chem-dust",))
    assert checks[0].status == "verified"
    assert "chem-dust" not in geog.default_geog_consumers()


def test_real_dust_fields():
    raw = os.environ.get("WOOF_DUST_GEOG")
    if not raw:
        pytest.skip("set WOOF_DUST_GEOG to the dust and land-use geog root")
    root = Path(raw)
    # The specs a preparation builds (woof.core.chem_statics), not a copy:
    # the rows' static sources resolved against this root's land use.
    from woof.chem_table import catalog
    from woof.core.chem_statics import static_specs
    from woof.static.build import GeogSelection
    specs = static_specs(catalog(), GeogSelection.fallback(root))
    required = [row["dataset_path"] for row in specs] + [
        row["water_mask"]["dataset_path"] for row in specs if "water_mask" in row]
    if not all((root / name / "index").is_file() for name in required):
        pytest.skip("dust or land-use datasets absent")
    if rust_bridge.unavailable_reason() is not None:
        pytest.skip("built Rust library absent")
    try:
        rust_bridge.require_extra_fields()
    except rust_bridge.StaticBridgeError:
        pytest.skip("library predates extra field contract")
    from woof.static.lambert import LambertGrid
    for lat, lon in [(33.5, -112.0), (25.0, -145.0)]:
        grid = LambertGrid(ref_lat=lat, ref_lon=lon, truelat1=30.0,
                           truelat2=60.0, stand_lon=lon, dx=12000.0,
                           dy=12000.0, e_we=101, e_sn=81)
        fields = build_extra_fields(grid, root, specs)
        assert fields["EROD"].shape == (3, 80, 100)
        for name, field in fields.items():
            assert field.dtype == np.float64 and np.isfinite(field).all()
            assert field.min() >= 0 and field.max() <= 1
            if lon == -145:
                assert not np.any(field), name
        if lon == -112:
            assert np.any(fields["EROD"] > 0)


def test_a_dataset_linked_into_the_root_is_accepted_and_climbing_out_is_not(tmp_path):
    """fetch-geog stages a root whose datasets are links into a shared
    WPS_GEOG tree.  Resolving before the containment check refused every
    such root (seen on a hrrr-prs chem preparation: 'dataset_path must name
    a directory under geog_root: modis_landuse_20class_30s_with_lakes')."""
    import pytest

    from woof.static.extra_fields import _dataset

    shared = tmp_path / "WPS_GEOG" / "landuse"
    shared.mkdir(parents=True)
    (shared / "index").write_text("type = categorical\n")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "landuse").symlink_to(shared, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem cannot make a directory link")
    assert _dataset(root.resolve(), "landuse") == str(root.resolve() / "landuse")
    for bad in ("../WPS_GEOG/landuse", str(shared), "."):
        with pytest.raises(ValueError, match="under geog_root"):
            _dataset(root.resolve(), bad)
