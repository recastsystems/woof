"""Installed waves must find the required helper without a developer PATH."""
import pytest


def test_installed_station_helper_without_environment_or_path(tmp_path, monkeypatch):
    from woof.hex import engines, wave_stations

    package = tmp_path / "hexcore"
    library = package / "libexec"
    library.mkdir(parents=True)
    helper = library / engines.executable_name("wave_audit")
    helper.write_bytes(b"fixture")
    helper.chmod(0o755)
    monkeypatch.setattr(engines, "__file__", str(package / "engines.py"))
    monkeypatch.setattr(wave_stations, "__file__", str(package / "wave_stations.py"))
    # (Recast WOOF) the installed helpers live where woof.hex.helpers says (its packaged directory)
    from woof.hex import helpers
    monkeypatch.setattr(helpers, "PACKAGED_DIR", library)
    monkeypatch.delenv("WOOF_HEX_RW_WAVE_AUDIT", raising=False)
    monkeypatch.delenv("WOOF_HEX_RW_WAVE_PRODUCTS", raising=False)
    monkeypatch.setattr(wave_stations.shutil, "which", lambda _: pytest.fail("install needed PATH"))
    assert wave_stations.audit_binary() == helper


def test_missing_explicit_station_helper_refuses_instead_of_falling_back(tmp_path, monkeypatch):
    from woof.hex import wave_stations

    monkeypatch.setenv("WOOF_HEX_RW_WAVE_AUDIT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="named hex wave station helper"):
        wave_stations.audit_binary()


def test_the_gsl_drag_statistics_helper_comes_from_the_staged_helper_archive(tmp_path, monkeypatch):
    """(Recast WOOF) rw_hex_ugwp_oro resolves from the hex helper archive a PyPI install stages (woof.hex.helpers).
    THE BREAKAGE: a clean install of the built wheels stopped every `woof hex regional ... --physics-suite hrrrv5` on a
    generated mesh at its static ("rw_hex_ugwp_oro is not installed"): the lookup read only woof/hex/libexec."""
    from pathlib import Path
    from woof.hex import helpers, ugwp_oro

    staged = tmp_path / "rw_hex_ugwp_oro"
    staged.write_bytes(b"fixture")
    monkeypatch.delenv(ugwp_oro.BINARY_ENV, raising=False)
    monkeypatch.setattr(helpers, "on_demand", lambda name: name == "rw_hex_ugwp_oro")
    monkeypatch.setattr(helpers, "path", lambda name, **kw: staged)
    packaged = Path(ugwp_oro.__file__).resolve().with_name("libexec") / "rw_hex_ugwp_oro"
    assert ugwp_oro.binary() == (packaged if packaged.is_file() else staged)
