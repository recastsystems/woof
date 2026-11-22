"""An old record-only fetch ABI must fail before a hash or asset transfer."""

from pathlib import Path
import subprocess

import pytest

from woof import doctor, rustwx_fetch
from woof.native_wrf_distribution import _BRIDGE_ABI_MARKERS


OLD_ABI = (
    "gpuwm-rw-fetch-record-v1\tmode\tmode_reason\tsource\tgrib_url\t"
    "idx_url\tidx_sha256\tidx_record_count\tselected_record_count\t"
    "ranges\tsha256"
)


def _binary_answers(monkeypatch, abi):
    monkeypatch.setattr(rustwx_fetch, "launchable", lambda path: (True, "ELF"))

    def answer(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, "rw_fetch 0.1.0\n" if args[-1] == "--version" else abi + "\n", "")
    monkeypatch.setattr(rustwx_fetch.subprocess, "run", answer)


def test_old_record_abi_refused_with_named_commands_and_remedy(monkeypatch):
    _binary_answers(monkeypatch, OLD_ABI)
    ok, reason = rustwx_fetch.probe_fetch_bin(Path("old-rw_fetch"))
    assert not ok, reason
    for name in ("rw_fetch", "hash", "asset", "inventory", "objects", "rebuild", "cargo build"):
        assert name in reason, reason


def test_doctor_uses_the_command_refusal(monkeypatch):
    _binary_answers(monkeypatch, OLD_ABI)
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: Path("old-rw_fetch"))
    result = doctor._fetch_backbone_check()
    assert result.status == "missing"
    assert "hash" in result.detail
    assert "rebuild" in result.detail


def test_packager_requires_commands_and_new_probe_accepts_them(monkeypatch):
    marker = _BRIDGE_ABI_MARKERS["rw_fetch"]
    assert marker == rustwx_fetch.FETCH_ABI_MARKER.encode("ascii")
    assert marker not in OLD_ABI.encode("ascii")
    _binary_answers(monkeypatch, marker.decode("ascii"))
    assert rustwx_fetch.probe_fetch_bin(Path("new-rw_fetch"))[0]


# The "public-objects" door this test also drove was the local data
# assimilation's object fetcher, removed in gpuwm 3.0.3.
@pytest.mark.parametrize("door", ["static-hash"])
def test_default_command_doors_refuse_before_transfer(monkeypatch, door):
    from woof import geog_assets

    _binary_answers(monkeypatch, OLD_ABI)
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: Path("old-rw_fetch"))

    def transfer(*args, **kwargs):
        pytest.fail("a stale binary reached the data command before its ABI refusal")

    monkeypatch.setattr(rustwx_fetch, "_run", transfer)
    with pytest.raises((geog_assets.GeogFetchError, ValueError), match="rw_fetch.*hash"):
        geog_assets._rw_asset("hash", input=Path("unused"))
