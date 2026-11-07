"""The doctor checks the archive the mesh static builder will open."""

from woof import rustwx_static
from woof.hex import doctor


def test_geography_finding_tracks_missing_partial_and_complete_archives(tmp_path, monkeypatch):
    root = tmp_path / "geog"
    monkeypatch.setattr(rustwx_static, "geog_root_candidates", lambda: (root,))
    missing = doctor.check_geography()[0]
    assert missing.status == doctor.MISSING and missing.required
    assert missing.remedy == "woof fetch-geog --datasets mesh"
    root.mkdir()
    assert doctor.check_geography()[0].status == doctor.MISSING
    for name in rustwx_static.REQUIRED_GEOG_DATASETS:
        directory = root / name
        directory.mkdir()
        (directory / "index").touch()
    assert doctor.check_geography()[0].status == doctor.VERIFIED
