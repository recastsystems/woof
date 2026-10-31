import sys

import pytest

from woof.hex.command_names import command_name
from woof.hex.engines import refusal_text


@pytest.mark.parametrize("invoked,expected", [
    ("woof hex", "woof"),
    ("weather hex", "weather"),
    ("/opt/bin/gpuwm-hex", "woof"),
    ("woof", "woof"),
])
def test_static_refusal_uses_invoking_front_door(monkeypatch, invoked, expected):
    monkeypatch.setattr(sys, "argv", [invoked])
    message = ("no geography root. Pass --geog DIR, run `woof fetch-geog`, or put the archive here:\n"
               "  /srv/woof/WPS_GEOG\n"
               "  woof fetch-geog --datasets mesh, which stages every dataset this builder opens")
    rendered = refusal_text(message)
    assert f"`{expected} fetch-geog`" in rendered
    assert f"  {expected} fetch-geog --datasets mesh," in rendered
    assert "/srv/woof/WPS_GEOG" in rendered
    assert "no geography root" in rendered
    assert command_name(engine=True) == expected


def test_static_remedy_does_not_rewrite_prose(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["woof hex"])
    message = "or run fetch-geog; custom fetch-geog; /bin/gpuwm fetch-geog"
    assert refusal_text(message) == message
