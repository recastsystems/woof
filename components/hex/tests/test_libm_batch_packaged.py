"""The wheel carries woof.hex.native_libm_batch's bridge, addressed by the module's own source key, without a GPU.

THE BREAKAGE THIS PREVENTS (30 km globe over 4 cards, reference system box, 2026-10-27): the bridge was only ever built on first
use with rustc, which no run box carries, so every GNU float batch fell back to one ctypes call per element on the GIL
(the vertical velocity damping: ~25 s of each rank's dycore build at 30 km, and every new region mesh paid it).
"""
import ast
import hashlib
import platform
from pathlib import Path

import pytest

from woof.hex import native_libm_batch

ROOT = Path(__file__).resolve().parents[1]


def test_setup_names_the_bridge_by_the_runtime_source_key():
    module = ast.parse((ROOT / "../../woof/hex/native_libm_batch.py").read_text())
    source = next(ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "_SOURCE" for target in node.targets))
    key = hashlib.sha256((source + platform.machine() + platform.system()).encode()).hexdigest()
    assert key == native_libm_batch.source_key()
    assert native_libm_batch.packaged_name(key) == f"libhex_libm_batch-{key}.so"
    setup = (ROOT / "setup.py").read_text()
    assert "self.build_libm_batch(root)" in setup and 'f"libhex_libm_batch-{key}.so"' in setup


def test_the_packaged_bridge_is_loaded_before_any_build(tmp_path, monkeypatch):
    """With the wheel's file present no compiler is looked up and nothing is built."""
    libexec = Path(native_libm_batch.__file__).with_name("libexec")
    packaged = libexec / native_libm_batch.packaged_name(native_libm_batch.source_key())
    if not packaged.is_file():
        pytest.skip("this install carries no packaged libm batch bridge")
    monkeypatch.setenv("WOOF_HEX_NATIVE_CPU_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(native_libm_batch.shutil, "which", lambda name: pytest.fail(f"looked up {name}"))
    native_libm_batch._library.cache_clear()
    try:
        assert native_libm_batch._library() is not None
        assert not (tmp_path / "empty-cache").exists()
    finally:
        native_libm_batch._library.cache_clear()
