"""The wave data bridge is staged on demand from the release bundle.

Breakage this prevents (3.0.0 lakes cut, 2026-11-10): from the README's pip install, `woof go --waves on` stopped
with "rw_waves_data is required" because this door looked for the bridge and never asked for the on-demand fetch
that the coupled-preparation door beside it already used. CPU only: the ladder and the fetch are stubbed.
"""

import pytest

from woof import bridges
from woof.ocean.waves import data as waves_data


def test_absent_bridge_is_fetched_on_demand_then_found(monkeypatch, tmp_path):
    staged = tmp_path / bridges.executable_name("rw_waves_data")
    asked = []

    def find_artifact(env_var, filename, **_):
        assert env_var == "WOOF_RW_WAVES_DATA"
        return staged if asked else None

    monkeypatch.setattr(bridges, "find_artifact", find_artifact)
    monkeypatch.setattr(bridges, "fetch_on_demand", lambda artifact: asked.append(artifact) or True)
    assert waves_data.native_binary() == staged
    assert asked == ["rw_waves_data"]


def test_found_bridge_is_not_fetched(monkeypatch, tmp_path):
    staged = tmp_path / "rw_waves_data"
    monkeypatch.setattr(bridges, "find_artifact", lambda *a, **k: staged)
    monkeypatch.setattr(bridges, "fetch_on_demand", lambda artifact: pytest.fail("fetched a bridge that was found"))
    assert waves_data.native_binary() == staged


def test_failed_fetch_names_fetch_bridges(monkeypatch):
    monkeypatch.setattr(bridges, "find_artifact", lambda *a, **k: None)
    monkeypatch.setattr(bridges, "fetch_on_demand", lambda artifact: False)
    with pytest.raises(FileNotFoundError, match="woof fetch-bridges"):
        waves_data.native_binary()
