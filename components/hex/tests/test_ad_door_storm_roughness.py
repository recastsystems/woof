"""adjoint-test reads a storm measure against the forecast's own roughness (roughness_view): the 12 km rain rows of
0.6.0 (one RTX 5090) read UNRESOLVED, not FAIL; a smooth measure's rows still pass; the README's
measures are read as before."""
import argparse

from woof.hex import ad_doors

PRED = -5.7281879167739444e-06   # <x, M^T w> of the 12 km rain measure along the bump
THR = {"dot_rel": 2e-05, "fd_tangent_err": 0.01, "taylor_best_err": 0.01, "zero_nonzero_words": 0}

# (h, dJ) of the real model, read on the 12 km example's storm band (a step-size ladder run and a zero-step run)
RAIN = [(1.0, -4.519496388702193e-06), (0.3, -1.4788746267e-06), (0.1, -6.471931885216397e-07),
        (0.03, -1.9214704328e-07), (0.01, -9.344122803357529e-08), (1e-06, -7.615349176553111e-08),
        (-1e-06, 2.973978410168713e-08), (-0.01, 2.295323901270474e-08), (-0.03, 1.6474548625e-07),
        (-0.1, 5.399348193781998e-07), (-0.3, 1.3866150208e-06), (-1.0, 5.1549018921055065e-06)]


def _doc(rows, pred=PRED):
    out = []
    for h, dj in rows:
        ratio = dj / (h * pred)
        out.append({"h": h, "dJ": dj, "fd_quotient": dj / h, "ratio": ratio, "err": abs(ratio - 1.0)})
    return {"tangent_dir": {"x_MTw": pred, "w_Mx": pred, "rel": 2.07e-08}, "taylor": {"rows": out},
            "zero_direction": {"nonzero_words": 0}}


def test_rain_rows_read_unresolved_not_fail():
    doc = _doc(RAIN)
    old = ad_doors.adjoint_verdict(doc, THR)
    assert not old["checks"]["fd"]["pass"] and not old["checks"]["fd"].get("unresolved")   # the old FAIL
    new = ad_doors.adjoint_verdict(doc, THR, probe=1e-6)
    assert new["checks"]["fd"].get("unresolved") and new["checks"]["taylor"].get("unresolved")
    assert new["passed"]   # dot and zero pass; FD and Taylor are not readings either way
    assert ad_doors.failed_checks(new["checks"]) == []
    rough = new["roughness"]
    assert abs(rough["floor"] - 7.615349176553111e-08) < 1e-20
    hs = [r["h"] for r in new["taylor_rows"]]
    assert hs == [1.0, 0.3, 0.1, 0.03, 0.01]   # centred, probes left out
    centred = {r["h"]: r["ratio"] for r in new["taylor_rows"]}
    assert 1.03 < centred[0.1] < 1.04 and 1.03 < centred[0.03] < 1.04
    d = ad_doors.direction_reading(doc, 0.01, 0.1, 3, probe=1e-6)
    assert d["status"] == "unresolved" and "roughness" in d["why"]
    assert not any(r["counts"] for r in d["rows"])
    verdict = ad_doors.directions_verdict([{**d, "name": "bump"}], new["passed"], {"min_resolved": 2})
    assert verdict["verdict"] == "unresolved"


def test_smooth_measure_with_probe_still_reads():
    rows = [(h, PRED * h * (1.0 + 0.002 * h)) for h in (1.0, 0.3, 0.1, 0.03, 0.01, -0.01, -0.03, -0.1, -0.3, -1.0)]
    rows += [(1e-6, 1e-15), (-1e-6, -2e-15)]
    doc = _doc(rows)
    v = ad_doors.adjoint_verdict(doc, THR, probe=1e-6)
    assert v["checks"]["fd"]["pass"] and v["checks"]["taylor"]["pass"] and v["passed"]
    d = ad_doors.direction_reading(doc, 0.01, 0.1, 3, probe=1e-6)
    assert d["status"] == "inside"


def test_without_probe_rows_nothing_changes():
    doc = _doc([r for r in RAIN if abs(r[0]) > 1e-3])
    assert ad_doors.adjoint_verdict(doc, THR, probe=1e-6) == ad_doors.adjoint_verdict(doc, THR)
    assert ad_doors.direction_reading(doc, 0.01, 0.1, 3, probe=1e-6) == ad_doors.direction_reading(doc, 0.01, 0.1, 3)


def test_storm_ladder_and_probe_only_for_storm_measures():
    table = ad_doors.load_table()
    assert ad_doors.storm_measure_probe(table, argparse.Namespace(metric="qr")) == 1e-6
    assert ad_doors.storm_measure_probe(table, argparse.Namespace(metric="w")) == 1e-6
    assert ad_doors.storm_measure_probe(table, argparse.Namespace(metric="theta_m")) is None
    assert ad_doors.storm_taylor_h([1, 0.3, 0.1], 1e-6) == [1.0, 0.3, 0.1, 1e-6, -1e-6, -0.1, -0.3, -1.0]


def test_storm_ladder_of_the_12km_row():
    table = ad_doors.load_table()
    own = table["storm_measures"]["taylor_h"]["hrrrv5-q12.96.2890.n40.60w91.30"]
    hs = ad_doors.storm_taylor_h([1, 0.3, 0.1, 0.03, 0.01], 1e-6, own)
    assert hs[:3] == [1.0, 0.6, 0.3] and 1e-6 in hs and -1e-6 in hs and -0.6 in hs


# (h, dJ) of the 12 km updraft measure (w, levels 12-30), bump and descent, one measured run on 0.6.0
W_BUMP = (0.01534419567549925, [(1.0, 0.015454277836049048), (0.6, 0.009357288492007053), (0.3, 0.004642578633745295),
          (0.1, 0.001484307306513407), (0.03, 0.00043650800321393005), (1e-06, 2.639765405569003e-06),
          (-1e-06, 1.089755023772384e-05), (-0.03, -0.0004447136403236878), (-0.1, -0.0015337803148083098),
          (-0.3, -0.004605757442467105), (-0.6, -0.009046769373030339), (-1.0, -0.015393448035850596)])


def test_updraft_bump_reads_inside_centred():
    pred, rows = W_BUMP
    d = ad_doors.direction_reading(_doc(rows, pred), 0.01, 0.1, 3, probe=1e-6)
    assert d["status"] == "inside"
    assert [r["h"] for r in d["rows"] if r["counts"]] == [1.0, 0.6, 0.3]
