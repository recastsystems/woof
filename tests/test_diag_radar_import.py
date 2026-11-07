"""Radar output must survive stock import without enabling lightning."""
from test_namelist_import import _generic_hierarchy_pair


def test_stock_radar_without_lightning_reaches_run_config(tmp_path):
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment

    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    inp.write_text(inp.read_text().replace(
        "&physics", "&physics\n do_radar_ref=1, lightning_option=0,"))
    text, report = import_namelists(wps, inp, name="radar-output")
    target = tmp_path / "radar.toml"
    target.write_text(text)
    cfg = load_experiment(target).root.run
    assert cfg.do_radar_ref == 1
    assert cfg.lightning_option == 0
    assert "inert: lightning_option = 0 reads no lightning control" not in report.format()


def test_explicit_radar_off_preserves_default_import_bytes(tmp_path):
    from woof.namelist_import import import_namelists

    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    before, _ = import_namelists(wps, inp, name="radar-default")
    inp.write_text(inp.read_text().replace(
        "&physics", "&physics\n do_radar_ref=0, lightning_option=0,"))
    after, _ = import_namelists(wps, inp, name="radar-default")
    assert before == after
