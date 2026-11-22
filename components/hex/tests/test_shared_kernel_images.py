"""A forecast binds the box's kernel image cache, not its own scratch (woof.hex.cuda_backend.cupy_cache).

Brought from fork jacobbieker/woof PR 488 (Apache-2.0).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from woof.hex.cuda_backend import cupy_cache


def test_forecast_images_live_in_the_box_kernel_root(tmp_path, monkeypatch):
    monkeypatch.delenv(cupy_cache.SHARED_ENV, raising=False)
    monkeypatch.setenv("WOOF_HEX_KERNEL_CACHE_ROOT", str(tmp_path / "box"))
    # Two runs in fresh work dirs share one image cache, so the second compiles nothing again.
    assert cupy_cache.forecast_directory(tmp_path / "run-1" / "scratch") == tmp_path / "box" / "cupy"
    assert cupy_cache.forecast_directory(tmp_path / "run-2" / "scratch") == tmp_path / "box" / "cupy"


def test_the_default_box_root_is_the_kernel_stores(tmp_path, monkeypatch):
    from woof.hex.kernel_warm import cache_root

    for name in (cupy_cache.SHARED_ENV, "WOOF_HEX_KERNEL_CACHE_ROOT", "WOOF_HEX_CACHE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cupy_cache.forecast_directory(tmp_path / "scratch") == cache_root() / "cupy"
    assert Path(tmp_path / "xdg") in cupy_cache.forecast_directory(tmp_path / "scratch").parents


def test_switch_off_keeps_the_per_run_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv(cupy_cache.SHARED_ENV, "0")
    monkeypatch.setenv("WOOF_HEX_KERNEL_CACHE_ROOT", str(tmp_path / "box"))
    assert cupy_cache.forecast_directory(tmp_path / "scratch") == tmp_path / "scratch"


def test_an_unwritable_box_root_falls_back_to_the_scratch(tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    root = tmp_path / "box"
    root.mkdir()
    root.chmod(0o555)
    try:
        monkeypatch.delenv(cupy_cache.SHARED_ENV, raising=False)
        monkeypatch.setenv("WOOF_HEX_KERNEL_CACHE_ROOT", str(root))
        assert cupy_cache.forecast_directory(tmp_path / "scratch") == tmp_path / "scratch"
    finally:
        root.chmod(0o755)


def test_the_forecast_driver_binds_the_box_cache():
    """Through hex 0.6.1 the driver passed its --scratch straight to require_cuda and KernelCache."""
    import woof.hex.drivers.run_cuda_v841_forecast as driver

    text = Path(driver.__file__).read_text()
    assert "images = forecast_directory(cache_root)" in text
    assert "KernelCache(capability=capability, cache_dir=cache_root)" not in text
