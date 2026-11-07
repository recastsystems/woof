"""Sealed static content survives relocation without accepting changed operands."""

from dataclasses import replace
from datetime import date
import json

import pytest

from woof.hrrr_native_static import verify_hrrr_native_static
from woof.static import highres_production as owner
from test_hrrr_native_static import _fixture


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("form", ["echo", "identity"])
def test_prepared_settings_match_binds_both_receipt_forms(tmp_path, enabled, form):
    config = owner.HighresStaticConfig(enabled=enabled, cache_root=tmp_path)
    recorded = (config.echo() if form == "echo"
                else owner.static_highres_identity(config))
    recorded["cache_root"] = "/preparation/source-cache"
    assert owner.prepared_highres_settings_match(recorded, config)
    assert not owner.prepared_highres_settings_match(recorded, None)
    assert not owner.prepared_highres_settings_match(None, config)
    assert owner.prepared_highres_settings_match(None, None)
    assert not owner.prepared_highres_settings_match(
        {**recorded, "enabled": not enabled}, config)
    assert not owner.prepared_highres_settings_match(
        {**recorded, "enabled": int(enabled)}, config)
    assert not owner.prepared_highres_settings_match(
        {**recorded, "unknown_setting": "different"}, config)


def _sealed(tmp_path, *, status="APPLIED", on_refuse="error"):
    target, cache, receipt_path = _fixture(tmp_path)
    config = owner.HighresStaticConfig(
        enabled=True, cache_root=tmp_path / "fetch-cache", on_refuse=on_refuse)
    grid, day = target.grid(), date(2026, 9, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["highres"] = {
        "status": status, "config": config.echo(),
        "case_date": day.isoformat(), "grid": owner._grid_identity(grid, 1),
    }
    # A folder spelled by a different operating system stays in the receipt.
    receipt["highres"]["config"]["cache_root"] = "/preparation/source-cache"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return target, cache, receipt_path, config, grid, day


@pytest.mark.parametrize("status,on_refuse", [
    ("APPLIED", "error"),
])
def test_sealed_overlay_accepts_relocated_cache(tmp_path, status, on_refuse):
    target, cache, path, config, grid, day = _sealed(
        tmp_path, status=status, on_refuse=on_refuse)
    original = path.read_bytes()
    _, receipt = verify_hrrr_native_static(cache, path, target)
    owner.require_prepared_highres(
        receipt, grid, config=config, domain_id=1, case_date=day)
    assert path.read_bytes() == original
    assert not config.cache_root.exists()


def test_sealed_overlay_reuses_relocated_cache_without_fetch(tmp_path, monkeypatch):
    target, cache, path, config, grid, day = _sealed(tmp_path)
    fields, receipt = verify_hrrr_native_static(cache, path, target)
    monkeypatch.setattr(owner, "apply_highres_statics", lambda *a, **kw:
                        pytest.fail("verified overlay was fetched again"))
    reused, evidence = owner.apply_prepared_highres(
        fields, grid, config=config, domain_id=1, case_date=day,
        landuse_attrs=None, baseline_receipt=receipt)
    assert reused is fields
    assert evidence is receipt


@pytest.mark.parametrize("setting,value", [
    ("terrain_source", "copernicus-dem-glo30"),
    ("landcover_source", "cglc-modis-lcz"),
    ("fields", "terrain"), ("on_refuse", "fallback-30s"),
])
def test_relocated_overlay_still_rejects_changed_settings(tmp_path, setting, value):
    target, cache, path, config, grid, day = _sealed(tmp_path)
    _, receipt = verify_hrrr_native_static(cache, path, target)
    with pytest.raises(ValueError, match="do not bind"):
        owner.require_prepared_highres(
            receipt, grid, config=replace(config, **{setting: value}),
            domain_id=1, case_date=day)


def test_relocated_overlay_still_rejects_changed_payload(tmp_path):
    target, cache, path, _, _, _ = _sealed(tmp_path)
    payload = bytearray(cache.read_bytes())
    payload[-1] ^= 1
    cache.write_bytes(payload)
    with pytest.raises(ValueError, match="cache differs from its receipt"):
        verify_hrrr_native_static(cache, path, target)
