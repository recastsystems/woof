"""A run that names --native-physics-tables reads every TEMPO lookup file from those directories, or refuses.

THE BREAKAGE (box-4x, hrrrv5 x1.40962, 8 rank processes, fast, 2026-11-06): the strict lookup files were being copied
into a named directory while two runs started; both silently read the fork source's NOAA copies and differed from
every later run on the same tree by 228,559 history words at step 1.  With no directory named, the fork copies
stay the default.
"""
from __future__ import annotations

import inspect

import pytest

from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpasphys import resources_build, runtime_v842
from woof.hex.mpasphys.rrfs import framework_init_rrfs as fi

FILES = [row[0] for row in fi.FrameworkInitializationRRFS.TEMPO_LOOKUP_FILES]
CONFIGS = {"config_microp_scheme": "mp_tempo"}
SOURCE = {"resource": "mpas-source-rrfs-44c0849", "pin_set": "rrfs-44c0849"}


@pytest.fixture
def fork(monkeypatch, tmp_path):
    """The fork source tree; records every time a run resolves it."""
    calls = []
    tree = tmp_path / "fork"
    (tree / "src/core_atmosphere/physics/physics_wrf/files").mkdir(parents=True)

    def ensure(resources, *, pin_set):
        calls.append((tuple(resources), pin_set))
        return {name: tree for name in resources}
    monkeypatch.setattr(resources_build, "ensure_route_resources", ensure)
    return calls


def _roots(tmp_path, held):
    """Two named directories as a deck names them: the three big files in one, ccn_activate.bin in the other."""
    first, second = tmp_path / "strict-tables", tmp_path / "tempo-tables"
    first.mkdir(); second.mkdir()
    for name in held:
        (second if name == "ccn_activate.bin" else first).joinpath(name).write_bytes(name.encode())
    return [first, second]


def test_a_named_directory_missing_a_file_refuses_in_one_line(tmp_path, fork):
    roots = _roots(tmp_path, [name for name in FILES if name != "qr_acr_qg_data_tempo_v3"])
    with pytest.raises(ConfigurationRefusal) as refused:
        fi.tempo_lookup_path_in(roots, "qr_acr_qg_data_tempo_v3", **SOURCE)
    message = str(refused.value)
    assert "\n" not in message and "qr_acr_qg_data_tempo_v3" in message and str(roots[0]) in message
    assert fork == []                        # never filled from the fork source's copies


def test_the_read_ahead_refuses_naming_every_missing_file(tmp_path, fork):
    roots = _roots(tmp_path, ["qr_acr_qs_data_tempo_v3"])
    with pytest.raises(ConfigurationRefusal) as refused:
        fi.FrameworkInitializationRRFS.prefetch_tables(CONFIGS, roots, **SOURCE)
    message = str(refused.value)
    assert "\n" not in message
    for name in ("freeze_water_data_tempo_v3", "qr_acr_qg_data_tempo_v3", "ccn_activate.bin"):
        assert name in message
    assert "qr_acr_qs_data_tempo_v3" not in message.split("holds", 1)[1].split(", and a named", 1)[0]
    assert fork == []


def test_complete_named_directories_serve_every_file(tmp_path, fork):
    roots = _roots(tmp_path, FILES)
    paths = [fi.tempo_lookup_path_in(roots, name, **SOURCE) for name in FILES]
    assert paths == [(roots[1] if name == "ccn_activate.bin" else roots[0]) / name for name in FILES]
    assert fi.missing_named_tables(roots, FILES) is None
    assert fork == []


def test_no_named_directory_keeps_the_fork_copies(fork):
    path = fi.tempo_lookup_path_in(None, "freeze_water_data_tempo_v3", **SOURCE)
    assert path.name == "freeze_water_data_tempo_v3" and "physics_wrf/files" in path.as_posix()
    assert fork == [(("mpas-source-rrfs-44c0849",), "rrfs-44c0849")]
    assert fi.missing_named_tables(None, FILES) is None and fi.missing_named_tables([], FILES) is None


def test_the_refusal_comes_before_the_dycore_inputs():
    source = inspect.getsource(runtime_v842._run_native_dynamics)
    assert source.index("prefetch_tables(layout.configs, named_tables") < source.index("'dycore-inputs'")
