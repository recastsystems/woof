"""A storm measure (rain water, cloud water, vertical velocity) prints the row's measured line before it runs."""
import argparse

from woof.hex import ad_doors


def _args(**kw):
    base = {"row": None, "mesh": None, "metric": "theta_m"}
    base.update(kw)
    return argparse.Namespace(**base)


def test_storm_measure_line_on_the_example_rows():
    table = ad_doors.load_table()
    block = table["storm_measures"]
    assert set(block["metrics"]) == {"qr", "qc", "w"}
    for row in ("hrrrv5-q12.96.2890.n40.60w91.30", "hrrrv5-q3.96.38278.n40.60w91.30"):
        assert row in table["rows"] and row in block["rows"]
        for metric in ("qr", "w"):
            lines = ad_doors.storm_measure_lines(table, _args(row=row, metric=metric), "adjoint-test", [metric])
            assert len(lines) == 1 and "no PASS of this release covers it" in lines[0] and "LIMITS.md" in lines[0]
    twelve = ad_doors.storm_measure_lines(table, _args(row="hrrrv5-q12.96.2890.n40.60w91.30"), "sensitivity",
                                          ["qr", "w", "theta_m"])
    assert "13,000" in twelve[0] and "(qr, w)" in twelve[0]


def test_no_line_for_the_readme_measures():
    table = ad_doors.load_table()
    for metric in ("theta_m", "theta", "qv", None):
        assert ad_doors.storm_measure_lines(table, _args(row="hrrrv5-q12.96.2890.n40.60w91.30"), "sensitivity",
                                            [metric]) == []
    assert ad_doors.storm_measure_lines(table, _args(mesh="x1.2562", metric="theta_m"), "sensitivity", ["theta_m"]) == []


def test_a_named_mesh_says_it_has_no_reading():
    table = ad_doors.load_table()
    lines = ad_doors.storm_measure_lines(table, _args(mesh="x1.2562", metric="qr"), "sensitivity", ["qr"])
    assert len(lines) == 1 and "no PASS of this release covers it" in lines[0] and "a mesh named with --mesh (x1.2562)" in lines[0]


def test_a_row_without_a_reading_says_so():
    table = ad_doors.load_table()
    other = next(r for r in table["rows"] if r not in table["storm_measures"]["rows"])
    lines = ad_doors.storm_measure_lines(table, _args(row=other, metric="w"), "adjoint-test", ["w"])
    assert lines and "has not been read on row" in lines[0]


def test_param_grad_prints_the_line_before_it_runs(monkeypatch, capsys):
    """param-grad takes --metric too: a storm measure prints the row's line before the door builds anything."""
    import pytest

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    parser = argparse.ArgumentParser()
    ad_doors.add_ad_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["param-grad", "--row", "hrrrv5-q3.96.38278.n40.60w91.30", "--metric", "w",
                              "--out", "pg"])
    monkeypatch.setattr(ad_doors, "Context", stop)
    with pytest.raises(Stop):
        args.handler(args)
    out = capsys.readouterr().out
    assert "woof hex param-grad: storm measure (w): no PASS of this release covers it" in out
