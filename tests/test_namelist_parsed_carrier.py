"""Parsed namelists retain virtual source paths and strict real-file metadata."""
from pathlib import Path
import tomllib

import pytest

from woof.namelist_import import import_parsed_namelists
from test_namelist_tolerance import _registry_default_pair


@pytest.mark.parametrize("filename,solar", [("namelist.input", 0), ("hrrr_wrf.nl", 1)])
def test_parsed_virtual_path_keeps_filename_defaults_without_reading_a_file(tmp_path, filename, solar):
    wps, inp = _registry_default_pair()
    path = tmp_path / filename
    assert not path.exists()
    text, _ = import_parsed_namelists(wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path)
    assert tomllib.loads(text)["shared"].get("alb_sol", 0) == solar
    assert not path.exists()


def test_parsed_existing_path_still_refuses_unknown_carried_selector(tmp_path):
    wps, inp = _registry_default_pair()
    path = tmp_path / "namelist.input"
    path.write_text('! gpuwm-physics-selectors-v1: {"unknown_selector": "wrf_461"}\n')
    with pytest.raises(ValueError, match="unknown"):
        import_parsed_namelists(wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path)


def test_parsed_existing_path_carries_a_valid_selector_without_a_snapshot(tmp_path):
    wps, inp = _registry_default_pair()
    path = tmp_path / "namelist.input"
    path.write_text('! gpuwm-physics-selectors-v1: {"bl_mynn_version": "wrf_461"}\n')
    text, report = import_parsed_namelists(
        wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path)
    assert tomllib.loads(text)["shared"]["bl_mynn_version"] == "wrf_461"
    carried = [row for row in report.defaults_applied if row.key == "bl_mynn_version"]
    assert len(carried) == 1
    assert carried[0].reason == "scheme generation explicitly carried by the namelist comment"


def test_parsed_snapshot_is_authoritative_and_does_not_read_the_existing_label(
        tmp_path, monkeypatch):
    wps, inp = _registry_default_pair()
    path = tmp_path / "namelist.input"
    path.write_text('! gpuwm-physics-selectors-v1: {"unknown_selector": "wrf_461"}\n')
    read_text = Path.read_text

    def reject_label_read(self, *args, **kwargs):
        if self == path:
            pytest.fail("explicit source snapshot reread its provenance label")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_label_read)
    text, _ = import_parsed_namelists(
        wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path,
        source_text='! gpuwm-physics-selectors-v1: {"bl_mynn_version": "wrf_461"}\n')
    assert tomllib.loads(text)["shared"]["bl_mynn_version"] == "wrf_461"


def test_parsed_virtual_snapshot_still_refuses_unknown_carried_selector(tmp_path):
    wps, inp = _registry_default_pair()
    path = tmp_path / "namelist.input"
    assert not path.exists()
    with pytest.raises(ValueError, match="unknown"):
        import_parsed_namelists(
            wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path,
            source_text='! gpuwm-physics-selectors-v1: {"unknown_selector": "wrf_461"}\n')
    assert not path.exists()


def test_parsed_explicit_source_text_refuses_unknown_carried_selector(tmp_path):
    wps, inp = _registry_default_pair()
    path = tmp_path / "namelist.input"
    path.write_text('! gpuwm-physics-selectors-v1: {"unknown_selector": "wrf_461"}\n')
    with pytest.raises(ValueError, match="unknown"):
        import_parsed_namelists(
            wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path,
            source_text=path.read_text(encoding="utf-8"))


def test_parsed_empty_snapshot_keeps_the_provenance_label_independent(tmp_path):
    # 4e3480479, development branch, keeps snapshots independent
    # of labels. An explicit empty snapshot retains that rule while the
    # missing-snapshot existing-file refusal remains required at this gate.
    wps, inp = _registry_default_pair()
    path = tmp_path / "namelist.input"
    expected, _ = import_parsed_namelists(
        wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path,
        source_text="")
    path.write_text('! gpuwm-physics-selectors-v1: {"unknown_selector": "wrf_461"}\n')
    actual, _ = import_parsed_namelists(
        wps, inp, wps_path=tmp_path / "namelist.wps", input_path=path,
        source_text="")
    assert actual == expected
