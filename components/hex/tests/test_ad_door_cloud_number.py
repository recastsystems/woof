# SPDX-License-Identifier: Apache-2.0
"""The 3 km storm row's default cloud derivative lets rain drop number and cloud droplet number follow the forecast.

THE BREAKAGE THIS PREVENTS: with both numbers held (the gated images) the gradient of reflectivity is wrong in size:
on the row's example over 59 steps the predicted change of summed reflectivity read 1.25 to 1.46 times the real
model's change along a rain nudge, and 0.37 to 0.46 of it for echo over 35 dBZ on a 12 km storm cut over 15 steps.
With the numbers following it read 0.930 / 1.035 / 1.086. A table edit that put the held images back as the default,
or named images the pinned download does not hold, would bring that back silently."""
import json
from types import SimpleNamespace

from woof.hex import ad_doors as d
from woof.hex.ad import assets

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
TAG = "0x06-rd3600n100-nfr-nfc-ts-dg200"


def _row():
    return d.load_table()["rows"][STORM]


def _ns(**kw):
    base = dict(plan=None, init=None, boundary_layer=None, cloud_derivative=None, checked_forward=False, row=STORM,
                mesh=None, keep_forward=None, objectives=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_number_is_the_storm_rows_default_and_names_its_four_images():
    row = _row()
    cd = row["gradient_line"]["cloud_derivative"]
    assert cd["default"] == "number" and sorted(cd["routes"]) == ["gated", "number", "shipped"]
    name, route, default = d.cloud_derivative_route(row, None)
    assert (name, default) == ("number", True)
    shas = {k: v["sha256"][:16] for k, v in route["assets"].items()}
    assert shas == {"tempo_block_cubin": "e1d67c6a5bfc3991", "tempo_tlf_cubin": "bf6f507fff5b2419",
                    "tempo_cubin": "2b80d55bd7528061"}
    assert route["also_packed"]["sha256"][:16] == "6fb6043551d41db2"
    for taped in (None, "taped"):          # the default route and the exact route both take it
        fast, taken = d.route_row(row, _ns(boundary_layer=taped))
        assert fast["assets"]["tempo_block_cubin"].endswith(f"cloud-number/c2chain-{TAG}.cubin")
        assert fast["assets"]["tempo_tlf_cubin"].endswith(f"cloud-number/tlfd-{TAG}.cubin")
        assert fast["assets"]["tempo_cubin"].endswith(f"cloud-number/split2-{TAG}.cubin")
    gated, taken = d.route_row(row, _ns(cloud_derivative="gated"))
    assert "cloud-number" not in json.dumps(gated["assets"])


def test_the_number_file_is_pinned_and_fetched_by_default_only():
    pins = json.loads(assets.PINS.read_text())
    v1 = pins["assets"]["ad-aux-cloud-number-sm120-v1.tar.gz"]
    assert v1["kind"] == "ad-aux" and len(v1["sha256"]) == 64
    assert sorted(h.rsplit("/", 1)[1] for h in v1["holds"]) == sorted(
        ["LICENSE-Apache-2.0.txt", "NOTICE.txt", f"c2chain-{TAG}.cubin", f"split2-{TAG}.cubin", f"tlfd-{TAG}.cubin",
         f"tlfd64-{TAG}.cubin"])
    row = _row()
    assert "ad-aux-cloud-number-sm120-v1.tar.gz" in assets.wanted(pins, d.example_reads(row, _ns()))
    for word in ("gated", "shipped"):
        assert "ad-aux-cloud-number-sm120-v1.tar.gz" not in assets.wanted(
            pins, d.example_reads(row, _ns(cloud_derivative=word)))


def test_the_texts_state_the_reflectivity_readings_and_their_scope():
    cd = _row()["gradient_line"]["cloud_derivative"]
    says = cd["routes"]["number"]["says"]
    for words in ("0.930 / 1.035 / 1.086", "1.251 / 1.392 / 1.461", "Below a 5 per cent nudge",
                  "not steady at any nudge size", "--cloud-derivative gated"):
        assert words in says, words
    assert "not this build's runner, and not through this command" in cd["default_when"]
    # 0.6.1: the route was read through the command on the final builds of 0.6.0, so it is no longer listed as not yet read
    assert "this build's runner and this command on a card" not in cd["routes"]["number"]["measured"]["not_yet"]
    assert "3.65e-04 and 1.92e-03 from 1" in says and "(-1.50 roundings)" in says
    assert "--cloud-derivative number" in cd["routes"]["gated"]["says"]
    w = cd["routes"]["number"]["measured"]["window_12km"]     # the twelve directions, as read
    assert "+3.97" in w and "no one-signed drift (mean +0.18, standard error 0.54)" in w
    # the held readings were taken on an earlier form of the derivative, not on the gated images: the texts must say which
    assert "gated itself was not read on this measure" in cd["default_when"]
    assert "an earlier form of the derivative" in says and "measured on an earlier form of the derivative" in cd["routes"]["gated"]["says"]
