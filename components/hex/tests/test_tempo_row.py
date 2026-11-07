"""The selectable TEMPO row must bind its actual scalar inventory."""
import importlib.util
from pathlib import Path

import pytest

from woof.hex.physics_suites import resolve_physics
from woof.hex.species_row import species_row_for_scheme


def test_tempo_numbers_and_volume_never_enter_moisture_loading():
    row = species_row_for_scheme("mp_tempo")
    assert row.n_species() == 13
    assert row.n_mass_loading == 6
    assert row.mass_loading_names() == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert row.names()[-2:] == ("ng", "volg")
    assert row.engine_scheme == "tempo"
    assert row.declaration("ng").units == "kg^{-1}"
    assert row.declaration("volg").units == "m^3 kg^{-1}"


def test_candidate_selection_keeps_tempo_instead_of_substituting_another_scheme():
    result = resolve_physics("rrfs_v2_tempo_candidate", mesh="conus12km")
    assert result.selectors["config_microp_scheme"] == "mp_tempo"
    assert result.seam == "column_suite"
    assert result.column_profile["microphysics"] == "mp_tempo"
    assert result.column_profile["generation"]["tempo_ml_for_bl_nc"] is True
    assert result.column_profile["generation"]["tempo_surface_emission_before"] is True
    assert result.convection == "ntiedtke"


def test_native_ml_switch_reaches_the_selected_engine_generation(tmp_path):
    from woof.hex.physics_namelist import read_physics_namelist
    path = tmp_path / "namelist.atmosphere"
    path.write_text("&physics\n config_physics_suite='rrfs_v2_tempo_candidate'\n/\n"
                    "&physics_mp_tempo\n config_tempo_ml_for_bl_nc=.true.\n/\n")
    result = read_physics_namelist(path, mesh="conus12km")
    assert result.admitted
    assert result.physics.column_profile["generation"]["tempo_ml_for_bl_nc"] is True
    path.write_text(path.read_text().replace(".true.", ".false."))
    result = read_physics_namelist(path, mesh="conus12km")
    assert result.admitted
    assert result.physics.column_profile["generation"]["tempo_ml_for_bl_nc"] is False


def test_identical_numeric_selections_keep_each_suites_own_provenance():
    first = resolve_physics("rrfs_v2_tempo_candidate", mesh="conus12km")
    other = resolve_physics("hrrrv5", mesh="conus12km")
    again = resolve_physics("rrfs_v2_tempo_candidate", mesh="conus12km")
    assert first.column_profile == again.column_profile
    assert first.column_profile["id"] != other.column_profile["id"]
    assert first.column_profile["differences_from_shaped_after"]


def _engine_root_with_tempo() -> Path:
    """The installed engine's root, or a skip naming the engine TEMPO needs."""
    spec = importlib.util.find_spec("woof")
    if spec is None or spec.origin is None:
        pytest.skip("no woof engine is installed")
    root = Path(spec.origin).resolve().parents[1]
    if not (root / "woof/core/tempo_runtime.py").is_file():
        pytest.skip("the installed woof carries no TEMPO port (woof 2.8.8 or later)")
    return root


def test_factory_and_repeat_measurement_bind_the_same_selected_microphysics_files():
    # The engine's column suite imports its microphysics, which imports cupy
    # at module level (gpuwm 2.9.0 woof/core/microphysics.py), so loading the
    # factory needs the device stack; the conftest marks this test gpu.  THE
    # BREAKAGE THIS PREVENTS: the public CPU tier (no CuPy) failed here.
    pytest.importorskip("cupy")
    from woof.hex.cuda_suite_physics_v841 import _load_suite_factory, _measure_engine
    profile = resolve_physics("rrfs_v2_tempo_candidate", mesh="conus12km").column_profile
    _, root, initial = _load_suite_factory(_engine_root_with_tempo(),
                                         profile["selection"], "tempo")
    repeated = _measure_engine(root, profile["selection"], "tempo")
    assert dict(initial.manifest) == dict(repeated.manifest)
    assert "woof/core/tempo_runtime.py" in initial.manifest
