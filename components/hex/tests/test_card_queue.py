"""woof.hex.card_queue: the host card queue's variables under their public names, and a site's own prefix.

THE BREAKAGE THIS PREVENTS (dist48, 2026-11-15): the package read and documented the variable
names of one particular card queue, in 26 source package files and 9 wheel files. The public names are
``CARD_QUEUE_*``; a site whose queue exports another prefix says so once, and hex reads both.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from woof.hex import card_queue


@pytest.fixture(autouse=True)
def no_site(monkeypatch, tmp_path):
    # A mapping passed in is the whole environment, so a test's own dict finds the config file under the home
    # folder, not under XDG_CONFIG_HOME. THE BREAKAGE (15 Nov 2026): on a machine whose site names its prefix in
    # ~/.config/gpuwm-hex/card-queue.json (every home node), the standard-names test read that site's prefix.
    monkeypatch.delenv(card_queue.PREFIX_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))


def test_the_standard_names_are_read_and_nothing_else():
    env = {"CARD_QUEUE_CARDS": "2,3", "SITEQ_CARDS": "9"}
    assert card_queue.names(card_queue.CARDS, env) == ("CARD_QUEUE_CARDS",)
    assert card_queue.granted(env) == ["2", "3"]
    assert card_queue.get(card_queue.RELEASE, "", env) == ""
    assert card_queue.granted({}) == []


def test_a_site_prefix_from_the_environment_is_read_after_the_standard_name():
    env = {card_queue.PREFIX_ENV: "SITEQ_", "SITEQ_CARDS": "4 5", "SITEQ_MODE": "pool"}
    assert card_queue.names(card_queue.CARDS, env) == ("CARD_QUEUE_CARDS", "SITEQ_CARDS")
    assert card_queue.granted(env) == ["4", "5"]
    assert card_queue.get(card_queue.MODE, "", env) == "pool"
    env["CARD_QUEUE_CARDS"] = "1"
    assert card_queue.granted(env) == ["1"]


def test_a_site_prefix_from_the_config_file(tmp_path, monkeypatch):
    folder = tmp_path / "config" / "woof hex"
    folder.mkdir(parents=True)
    (folder / "card-queue.json").write_text(json.dumps({"env_prefix": "SITEQ_"}))
    monkeypatch.setenv("SITEQ_VRAM_MIB", "9000")
    assert card_queue.get(card_queue.VRAM_MIB) == "9000"
    # Set but empty in the environment: the standard names only (tests isolate themselves this way).
    monkeypatch.setenv(card_queue.PREFIX_ENV, "")
    assert card_queue.get(card_queue.VRAM_MIB) == ""


def test_put_and_drop_write_every_name_a_child_reads():
    env = {card_queue.PREFIX_ENV: "SITEQ_", "SITEQ_CARDS": "4,5", "CARD_QUEUE_CARDS": "4,5"}
    card_queue.put(env, card_queue.CARDS, "5")
    assert env["CARD_QUEUE_CARDS"] == env["SITEQ_CARDS"] == "5"
    card_queue.drop(env, card_queue.CARDS)
    assert "CARD_QUEUE_CARDS" not in env and "SITEQ_CARDS" not in env


@pytest.mark.parametrize("prefix", ["", "CARD_QUEUE_", "1BAD", "A-B_", "$X"])
def test_an_unusable_site_prefix_is_ignored(prefix):
    env = {card_queue.PREFIX_ENV: prefix}
    assert card_queue.names(card_queue.CARDS, env) == ("CARD_QUEUE_CARDS",)


def test_an_unknown_variable_is_refused():
    with pytest.raises(KeyError):
        card_queue.names("CARD")


def test_every_shipped_reader_goes_through_card_queue():
    """No module reads a card queue variable from the environment by name, so a site's prefix reaches them all."""
    root = Path(card_queue.__file__).resolve().parent
    reads = re.compile(r"""environ\b[^\n]{0,24}["']CARD_QUEUE_[A-Z]""")
    spelled = [str(p.relative_to(root)) for p in root.rglob("*.py")
               if p.name != "card_queue.py" and reads.search(p.read_text(encoding="utf-8"))]
    assert spelled == []
