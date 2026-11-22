"""woof hex render --overlays: the caller's rings and labels drawn over the graticule.

THE BREAKAGE THIS GUARDS: a hail map for a partner rings the reports it is judged against and names the towns; the
render door put only its own graticule on a map, so those maps had to leave the product path.
"""
import json

import pytest

from woof.hex import render_door


def test_overlays_join_the_graticule(tmp_path):
    graticule = tmp_path / "g.json"
    graticule.write_text(json.dumps({"lines": [{"points": [[40, -95], [41, -95]], "color": "#999999", "width": 1}],
                                     "labels": [{"lat": 40, "lon": -95, "text": "40N"}]}))
    extra = tmp_path / "x.json"
    extra.write_text(json.dumps({"points": [{"lat": 41.6, "lon": -93.6, "color": "#000000", "radius_px": 4,
                                             "width_px": 1, "shape": "circle"}],
                                 "labels": [{"lat": 41.59, "lon": -93.62, "text": "Des Moines"}]}))
    out = render_door.merge_overlays(graticule, extra, tmp_path)
    doc = json.loads(out.read_text())
    assert len(doc["lines"]) == 1 and [l["text"] for l in doc["labels"]] == ["40N", "Des Moines"]
    assert doc["points"][0]["lat"] == 41.6
    assert json.loads(render_door.merge_overlays(None, extra, tmp_path).read_text())["labels"][0]["text"] == "Des Moines"


def test_an_unknown_overlay_entry_is_refused(tmp_path):
    extra = tmp_path / "x.json"
    extra.write_text(json.dumps({"circles": []}))
    with pytest.raises(render_door.RenderDoorError, match="unknown entry"):
        render_door.merge_overlays(None, extra, tmp_path)
