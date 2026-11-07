"""The new native reference must reject altered code or licence text."""
import pytest

from tools.ntu_wrf471_oracle.build import MANIFEST, digest, verify
from tools.ntu_wrf461_oracle.f2cu import logical_lines


def test_source_pin_is_the_unmodified_public_module():
    assert MANIFEST["source_changes"] == []
    assert MANIFEST["module"]["sha256"] == (
        "c864483fa572d42ab1dc51cc538f8062dc2f582369f597261e8c850c49ce34dd")
    assert MANIFEST["module"]["url"].startswith(
        "https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/")


def test_altered_licence_is_rejected_before_source_is_read(tmp_path):
    licence = tmp_path / "LICENSE.txt"
    licence.write_text("unknown licence")
    with pytest.raises(ValueError, match="LICENSE.txt: expected"):
        verify(tmp_path / "source-was-never-read.F", licence)


def test_altered_source_is_rejected(tmp_path, monkeypatch):
    licence, source = tmp_path / "LICENSE.txt", tmp_path / "module_mp_ntu.F"
    licence.write_text("test licence")
    source.write_text("altered source")
    monkeypatch.setitem(MANIFEST["licence"], "sha256", digest(licence))
    with pytest.raises(ValueError, match="module_mp_ntu.F: expected"):
        verify(source, licence)


def test_native471_single_precision_preprocessor_branch():
    source = """#ifdef DOUBLE_PRECISION
GAMLN = LOG_GAMMA(XX)
#else
GAMLN = SNGL(LOG_GAMMA(XX))
#endif
"""
    assert [statement for _, _, statement in logical_lines(source)] == [
        "GAMLN = SNGL(LOG_GAMMA(XX))"]


def test_nested_preprocessor_branches():
    source = """#ifndef DOUBLE_PRECISION
#if (RWORDSIZE == 4)
X = 1
#else
X = 2
#endif
#endif
"""
    assert [statement for _, _, statement in logical_lines(source)] == ["X = 1"]
