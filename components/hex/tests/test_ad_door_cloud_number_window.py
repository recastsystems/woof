# SPDX-License-Identifier: Apache-2.0
"""The number cloud route needs the transport limiter live: a long window runs the gated route by default and refuses
the number route by name (refl-row, 0.6.0)."""
import argparse
import json
from pathlib import Path

import pytest

import woof.hex
from woof.hex import ad_doors

ROW = "hrrrv5-q3.96.38278.n40.60w91.30"


def _table():
    return json.loads((Path(woof.hex.__file__).parent / "data" / "ad-doors.v1.json").read_text())


def _args(steps, cloud=None):
    return argparse.Namespace(row=ROW, mesh=None, steps=steps, cloud_derivative=cloud, checked_forward=False,
                              keep_forward=None, boundary_layer=None)


def test_the_number_route_needs_the_live_limiter_and_the_row_runs_it_live():
    row = _table()["rows"][ROW]
    w = row["window_args"]
    assert w[w.index("--fct-frozen") + 1] == "0"
    cd = row["gradient_line"]["cloud_derivative"]
    assert cd["default"] == "number" and cd["routes"]["number"]["needs_fct_live"]
    assert cd["past_live_route"] == "gated" and "needs_fct_live" not in cd["routes"]["gated"]


def test_a_window_inside_the_live_limiter_keeps_the_number_route():
    a = _args(59)
    ad_doors.cloud_route_for_window(a, _table())
    assert a.cloud_derivative is None


def test_a_longer_window_runs_gated_by_default_and_says_so(capsys):
    t = _table()
    steps = int(t["rows"][ROW]["fct_live_max_s"] // t["rows"][ROW]["dt_s"]) + 1
    a = _args(steps)
    ad_doors.cloud_route_for_window(a, t)
    assert a.cloud_derivative == "gated"
    assert "cloud derivative gated for this window" in capsys.readouterr().out


def test_the_number_route_by_name_on_a_longer_window_is_refused():
    t = _table()
    steps = int(t["rows"][ROW]["fct_live_max_s"] // t["rows"][ROW]["dt_s"]) + 1
    with pytest.raises(ad_doors.AdDoorError, match="--cloud-derivative gated runs it"):
        ad_doors.cloud_route_for_window(_args(steps, "number"), t)
    a = _args(steps, "gated")
    ad_doors.cloud_route_for_window(a, t)
    assert a.cloud_derivative == "gated"
